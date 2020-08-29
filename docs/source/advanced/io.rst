Loading and Saving
==================

Being able to load and save the weights of a model, or a model itself (e.g. the weights and the function itself)
is essential for machine learning purposes.
In this section we describe how to load/save the weights and also how to save an entire model.
Furthermore we discuss how to keep multiple saves, a concept known as checkpointing, which is typically used for
resuming interrupted training sessions.


Saving and loading model weights
--------------------------------

Loading and saving is done on :py:class:`objax.VarCollection` objects.
Such objects are returned by the :py:meth:`objax.Module.vars` method or can be constructed manually if one wishes to.
The saving method uses
`numpy .npz format <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`_ which in essence stores
tensors in a zip file.

Here's a simple example::

    import objax

    # Let's pretend we have a neural network net and we want to save it.
    net = objax.nn.Sequential([objax.nn.Linear(768, 1), objax.functional.sigmoid])

    # Saving only takes one line.
    objax.io.save_var_collection('net.npz', net.vars())

    # Let's modify the bias of the Linear layer
    net[0].b.assign(net[0].b.value + 1)
    print(net[0].b.value.sum())         # 1.0

    # Loading
    objax.io.load_var_collection('net.npz', net.vars())
    print(net[0].b.value.sum())         # 0.0

Note that in the example above we used a filename to specify where to save the weights. These APIs also accept a file
descriptor, so another way to save would be::

    # Saving with file descriptor
    with open('net.npz', 'wb') as f:
        objax.io.save_var_collection(f, net.vars())

    # Loading with file descriptor
    with open('net.npz', 'rb') as  f:
        objax.io.load_var_collection(f, net.vars())

.. note::
    The advantage of using a filename instead of file handle is that data will be written to a temporary file
    first and a temporary file will be renamed to provided filename only after all data has been written.
    In the event of the program being killed, this prevents from having truncated files.
    When using a file descriptor the code does not have this protection.
    File descriptors are typically used for unit testing.

Custom saving and loading
^^^^^^^^^^^^^^^^^^^^^^^^^

You can make your own saving and loading functions easily.
In essence saving has to store pairs of :code:`(name, numpy array)`, loading must provide a numpy array for the
variables of the :py:class:`objax.VarCollection`.
The only gotcha to pay attention to is to avoid saving duplicated information such as shared weights under different
names or variable references TrainRef.
Since the code for loading and saving is very concise, simply looking at it is the best example.

Checkpointing
-------------

Checkpointing can be defined as saving neural network weights during training.
Often checkpointing keeps multiple saves, each from different training steps.
For space reasons, it's common to keep only the latest-k saves.
Checkpointing can be used for a variety of purposes:

* Resuming training after the program was interrupted.
* Keeping multiple copies of the network for weight averaging strategies.

Objax provides a simple checkpointing interface called :py:class:`objax.io.Checkpoint`, here's an example::

    import objax

    # Let's pretend we have a neural network net and we want to save it.
    net = objax.nn.Sequential([objax.nn.Linear(768, 1), objax.functional.sigmoid])

    # This time we use the Checkpoint class
    ckpt = objax.io.Checkpoint(logdir='save_folder', keep_ckpts=5)

    # Saving
    ckpt.save(net.vars(), idx=1)
    net[0].b.assign(net[0].b.value + 1)
    ckpt.save(net.vars(), idx=2)

    # Restoring
    ckpt.restore(net.vars(), idx=1)   # net[0].b.value = (0,)
    ckpt.restore(net.vars(), idx=2)   # net[0].b.value = (1,)

    # When no epoch is specified use latest checkpoint (e.g. 2 here)
    idx, file = ckpt.restore(net.vars())
    print(idx, file)  # 2 save_folder/ckpt/0000000002.npz

Customized checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`objax.io.Checkpoint` class has some constants that allow it to customize its behavior.
You can redefine them for example creating a child class that inherits from Checkpoint.
The fields are the following::

    class Checkpoint:
        DIR_NAME: str = 'ckpt'
        FILE_MATCH: str = '*.npz'
        FILE_FORMAT: str = '%010d.npz'
        LOAD_FN: Callable[[FileOrStr, VarCollection], None] = staticmethod(load_var_collection)
        SAVE_FN: Callable[[FileOrStr, VarCollection], None] = staticmethod(save_var_collection)

This lets you change the folder name where the checkpoints are saved, the file extension and the numbering format.
If you have your own saving and loading functions, you can also replace them.
Remember to wrap them in :code:`staticmethod` since they don't depend on the Checkpoint class itself.

Saving a module
---------------

.. warning::
    `Python pickle is not *security* safe <https://www.google.com/search?q=pickle+remote+code+execution>`_.
    Only use it for your own saves and loads. Any pickle coming from an external source is a
    potential risk.

Now that we warned you, let's mention that Objax modules can be pickled
with `Python's pickle module <https://docs.python.org/3/library/pickle.html>`_ like many other Python objects.
This can be quite convenient since you can save not only the module's weight, but the module itself.

Let's look at a simple example::

    import pickle
    import objax

    # Let's pretend we have a neural network net and we want to save it as whole.
    net = objax.nn.Sequential([objax.nn.Linear(768, 1), objax.functional.sigmoid])

    # Pickling
    pickle.dump(net, open('net.pickle', 'wb'))

    # Unpickling and storing into a new network
    net2 = pickle.load(open('net.pickle', 'rb'))

    # Confirm the network net2 has the same function as net
    x = objax.random.normal((100, 768))
    print(((net(x) - net2(x)) ** 2).mean())  # 0.0

    # Confirm the network net2 does not share net's weights
    net[0].b.assign(net[0].b.value + 1)
    print(((net(x) - net2(x)) ** 2).mean())  # 0.038710583

As the example shows, pickling is really easy to use. Be aware that Python pickling has some limitations, namely lambda
functions cannot always be saved (they have to be named). Objax is not limited to pickle, since its design is pythonic
it should be compatible with other python pickling systems.
