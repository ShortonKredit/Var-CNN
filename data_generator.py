import h5py
import numpy as np
import threading


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe.

    Does this by serializing call to the `next` method of given iterator/
    generator. See https://anandology.com/blog/using-iterators-and-generators/
    for more information.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):  # Py2
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """Decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


def generate(config, data_type, mixture_num):
    """Yields batch of data with the correct content and formatting.

    Args:
        data_type (str): Either 'training_data', 'validation_data', or
            'test_data'
        config (dict): Deserialized JSON config file (see config.json)
    """

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst = num_mon_inst_train + num_mon_inst_test
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites_test = config['num_unmon_sites_test']

    data_dir = config['data_dir']
    batch_size = config['batch_size']
    mixture = config['mixture']
    use_dir = 'dir' in mixture[mixture_num]
    use_time = 'time' in mixture[mixture_num]
    use_metadata = 'metadata' in mixture[mixture_num]
    use_dir_iat_log = 'dir_iat_log' in mixture[mixture_num]
    use_dir_x_iat = 'dir_x_iat' in mixture[mixture_num]
    use_dir_iat_raw = 'dir_iat_raw' in mixture[mixture_num]

    # Load all data into memory first so h5py file can be closed
    data_file = config.get('processed_h5')
    if not data_file:
        data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites, num_mon_inst,
                                          num_unmon_sites_train, num_unmon_sites_test)

    with h5py.File(data_file, 'r') as f:
        dir_seq = f[data_type + '/dir_seq'][:] if 'dir_seq' in f[data_type] else None
        time_seq = f[data_type + '/time_seq'][:] if 'time_seq' in f[data_type] else None
        metadata = f[data_type + '/metadata'][:] if 'metadata' in f[data_type] else None
        labels = f[data_type + '/labels'][:]
        
        # Read new keys directly from H5 if requested
        if use_dir_iat_log:
            if 'dir_iat_log' in f[data_type]:
                dir_iat_log = f[data_type + '/dir_iat_log'][:]
            else:
                raise ValueError(f"dir_iat_log requested but not found in {data_file}")
                
        if use_dir_x_iat:
            if 'dir_x_iat' in f[data_type]:
                dir_x_iat = f[data_type + '/dir_x_iat'][:]
            else:
                raise ValueError(f"dir_x_iat requested but not found in {data_file}")
                
        if use_dir_iat_raw:
            if 'dir_iat_raw' in f[data_type]:
                dir_iat_raw = f[data_type + '/dir_iat_raw'][:]
            else:
                raise ValueError(f"dir_iat_raw requested but not found in {data_file}")

    batch_start = 0
    
    # --- SHUFFLE DATA TO PREVENT CATASTROPHIC FORGETTING ---
    indices = np.arange(len(labels))
    if data_type != 'test_data':
        np.random.shuffle(indices)

    while True:
        if batch_start >= len(labels):
            batch_start = 0
            
            # Shuffle again at the end of each epoch
            if data_type != 'test_data':
                np.random.shuffle(indices)

        batch_end = batch_start + batch_size
        batch_indices = indices[batch_start:batch_end]

        inputs = {}

        # Accesses and stores relevant data slices
        if use_dir:
            inputs['dir_input'] = dir_seq[batch_indices]
        if use_time:
            inputs['time_input'] = time_seq[batch_indices]
        if use_metadata:
            inputs['metadata_input'] = metadata[batch_indices]
        if use_dir_iat_log:
            inputs['dir_iat_log_input'] = dir_iat_log[batch_indices]
        if use_dir_x_iat:
            inputs['dir_x_iat_input'] = dir_x_iat[batch_indices]
        if use_dir_iat_raw:
            inputs['dir_iat_raw_input'] = dir_iat_raw[batch_indices]

        labels_batch = labels[batch_indices]
        batch_start += batch_size

        # Test data does not use labels
        if data_type == 'test_data':
            yield (inputs,)
        else:
            yield (inputs, labels_batch)

