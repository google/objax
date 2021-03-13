# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__all__ = ['find_used_variables']

import inspect
import ast
import re
import tokenize

from io import StringIO

from typing import Callable
from objax.variable import BaseVar, VarCollection
from objax.module import Module


def getanno(node, key, field_name='__anno'):
    """Gets annotation of the AST node."""
    return getattr(node, field_name)[key]


def hasanno(node, key, field_name='__anno'):
    """Returns whether AST node has an annotation."""
    return hasattr(node, field_name) and key in getattr(node, field_name)


def setanno(node, key, value, field_name='__anno'):
    """Sets annotation on AST node."""
    annotations = getattr(node, field_name, {})
    setattr(node, field_name, annotations)
    annotations[key] = value


class AnalyzeUserVariablesNodeTransformer(ast.NodeTransformer):

    def __init__(self, closure_vars, global_vars):
        self.closure_vars = closure_vars
        self.global_vars = global_vars
        self.vc = VarCollection()

    def check_objax_var_module(self, node):
        if not hasanno(node, 'value'):
            return
        v = getanno(node, 'value')
        v_name = getanno(node, 'name')
        if v is None:
            return
        if isinstance(v, Module):
            self.vc.update(v.vars(scope=v_name + '.'))
            setanno(node, 'value', None)
        if isinstance(v, BaseVar):
            if v_name in self.vc and self.vc[v_name] is not v:
                # This generally should not happen and probably indication of a bug somewhere.
                raise ValueError(
                    f'Variable tracing failed because two variables were found with the same name {v_name}')
            else:
                self.vc[v_name] = v
                setanno(node, 'value', None)

    def visit_Name(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            if node.id in self.closure_vars:
                setanno(node, 'name', node.id)
                setanno(node, 'value', self.closure_vars[node.id])
                self.check_objax_var_module(node)
            elif node.id in self.global_vars:
                setanno(node, 'name', node.id)
                setanno(node, 'value', self.global_vars[node.id])
                self.check_objax_var_module(node)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and hasanno(node.value, 'value'):
            parent_value = getanno(node.value, 'value')
            if parent_value is not None and hasattr(parent_value, node.attr):
                setanno(node, 'name', getanno(node.value, 'name') + '.' + node.attr)
                setanno(node, 'value', getattr(parent_value, node.attr))
                self.check_objax_var_module(node)

        return node


_LEADING_WHITESPACE = re.compile(r'\s*')


def dedent_block(code_string):
    """Dedents a code so that its first line starts at row zero."""

    # Removes any backslash line continuations from the code
    code_string = code_string.replace('\\\n', '')

    token_gen = tokenize.generate_tokens(StringIO(code_string).readline)

    block_indentation = None
    tokens = []
    try:
        for tok in token_gen:
            tokens.append(tok)
    except tokenize.TokenError:
        # Resolution of lambda functions may yield incomplete code, which can
        # in turn generate this error. We silently ignore this error because the
        # parser may still be able to deal with it.
        pass

    for tok in tokens:
        tok_type, tok_string, _, _, _ = tok
        if tok_type == tokenize.INDENT:
            block_indentation = tok_string
            break
        elif tok_type not in (tokenize.NL, tokenize.NEWLINE, tokenize.STRING, tokenize.COMMENT):
            block_indentation = ''
            break

    if not block_indentation:
        return code_string

    block_level = len(block_indentation)
    first_indent_uses_tabs = '\t' in block_indentation
    for i, tok in enumerate(tokens):
        tok_type, tok_string, _, _, _ = tok
        if tok_type == tokenize.INDENT:
            if ((' ' in tok_string and first_indent_uses_tabs) or ('\t' in tok_string and not first_indent_uses_tabs)):
                raise ValueError('Code mixing tabs and spaces for indentation is not allowed')
            if len(tok_string) >= block_level:
                tok_string = tok_string[block_level:]
            tokens[i] = (tok_type, tok_string)

    new_code = tokenize.untokenize(tokens)

    # Note: untokenize respects the line structure, but not the whitespace within
    # lines. For example, `def foo()` may be untokenized as `def foo ()`
    # So instead of using the output of dedent, we match the leading whitespace
    # on each line.
    dedented_code = []
    for line, new_line in zip(code_string.split('\n'), new_code.split('\n')):
        original_indent = re.match(_LEADING_WHITESPACE, line).group()
        new_indent = re.match(_LEADING_WHITESPACE, new_line).group()
        if len(original_indent) > len(new_indent):
            dedented_line = line[len(original_indent) - len(new_indent):]
        else:
            dedented_line = line
        dedented_code.append(dedented_line)
    new_code = '\n'.join(dedented_code)

    return new_code


def find_used_variables(fn: Callable) -> VarCollection:
    """Finds all Objax variables which are used by a given callable.

    Args:
        fn: input function or callable.

    Returns:
        Variable collection with all variables used by input function.
    """
    if not hasattr(fn, '__code__'):
        raise ValueError('Can not determine variables used by a function. Function does not have __code__ attribute.')

    try:
        src = inspect.getsource(fn)
    except OSError:
        raise ValueError('Can not determine variables used by a function. Code of the function can not be retrieved.')

    src = dedent_block(src)

    main_node = ast.parse(src)
    main_node = main_node.body[0]

    if fn.__closure__:
        closure_vars = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
    else:
        closure_vars = {}
    analyzer = AnalyzeUserVariablesNodeTransformer(closure_vars=closure_vars,
                                                   global_vars=fn.__globals__)
    analyzer.visit(main_node)

    return analyzer.vc
