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


def safe_h5_read(dataset, indices):
    """
    Safely reads arbitrary indices from h5py dataset.
    Sorts indices to satisfy h5py requirement of increasing index access,
    reads from dataset, and then restores the original requested index order.
    """
    sort_idx = np.argsort(indices)
    sorted_indices = indices[sort_idx]
    
    # Query H5 using sorted indices
    data = dataset[sorted_indices]
    
    # Unsort the retrieved data back to the requested order
    unsort_idx = np.argsort(sort_idx)
    return data[unsort_idx]


@thread_safe_generator
def generate(config, data_type, mixture_num=None):
    """Yields batch of data with the correct content and formatting.

    Args:
        config (dict): Deserialized JSON config file (see config.json)
        data_type (str): Either 'training_data', 'validation_data', or
            'test_data'
        mixture_num (int, optional): Index of the mixture in the config
    """
    batch_size = config['batch_size']
    data_file = config.get('processed_h5')
    if not data_file:
        num_mon_sites = config['num_mon_sites']
        num_mon_inst_train = config['num_mon_inst_train']
        num_mon_inst_test = config['num_mon_inst_test']
        num_mon_inst = num_mon_inst_train + num_mon_inst_test
        num_unmon_sites_train = config['num_unmon_sites_train']
        num_unmon_sites_test = config['num_unmon_sites_test']
        data_dir = config['data_dir']
        data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites, num_mon_inst,
                                          num_unmon_sites_train, num_unmon_sites_test)

    # Determine sequence and metadata configuration
    if "sequence_dataset" in config:
        # New flat config style
        seq_ds_name = config["sequence_dataset"]
        seq_input_name = config["sequence_input_name"]
        meta_ds_name = config.get("metadata_dataset")
        meta_type = config.get("metadata_type")
        wfmeta_k = config.get("wfmeta_k", 10)
    else:
        # Backward compatibility fallback
        if mixture_num is None:
            raise ValueError("Either mixture_num must be provided or sequence_dataset must be in config.")
        mixture = config['mixture']
        inner_comb = mixture[mixture_num]
        
        # Mapping old names
        seq_ds_name = None
        seq_input_name = None
        if 'dir' in inner_comb:
            seq_ds_name = 'dir_seq'
            seq_input_name = 'dir_input'
        elif 'time' in inner_comb:
            seq_ds_name = 'time_seq'
            seq_input_name = 'time_input'
        elif 'dir_iat_log' in inner_comb:
            seq_ds_name = 'dir_iat_log'
            seq_input_name = 'dir_iat_log_input'
        elif 'dir_x_iat' in inner_comb:
            seq_ds_name = 'dir_x_iat'
            seq_input_name = 'dir_x_iat_input'
        elif 'dir_iat_raw' in inner_comb:
            seq_ds_name = 'dir_iat_raw'
            seq_input_name = 'dir_iat_raw_input'
            
        meta_ds_name = 'metadata' if 'metadata' in inner_comb else None
        meta_type = 'metadata'
        wfmeta_k = 10

    # Stream data from H5 file batch by batch
    with h5py.File(data_file, 'r') as f:
        grp = f[data_type]
        
        seq_ds = grp[seq_ds_name] if (seq_ds_name and seq_ds_name in grp) else None
        meta_ds = grp[meta_ds_name] if (meta_ds_name and meta_ds_name in grp) else None
        labels_ds = grp['labels']
        
        num_samples = len(labels_ds)
        indices = np.arange(num_samples)
        
        # Shuffle indices for training/validation (not for test_data)
        if data_type != 'test_data':
            np.random.shuffle(indices)

        batch_start = 0
        while True:
            if batch_start >= num_samples:
                batch_start = 0
                if data_type != 'test_data':
                    np.random.shuffle(indices)

            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_start += batch_size

            inputs = {}
            
            # Load sequence batch
            if seq_ds is not None:
                seq_batch = safe_h5_read(seq_ds, batch_indices)
                inputs[seq_input_name] = seq_batch.astype(np.float32)

            # Load metadata batch
            if meta_ds is not None:
                meta_batch = safe_h5_read(meta_ds, batch_indices)
                if meta_type == "wfmeta10" or meta_ds_name == "wfmeta":
                    # Extract top-k ANOVA features (defaults to 10)
                    meta_batch = meta_batch[:, :wfmeta_k]
                inputs['metadata_input'] = meta_batch.astype(np.float32)

            labels_batch = safe_h5_read(labels_ds, batch_indices).astype(np.float32)

            # Test data does not yield labels
            if data_type == 'test_data':
                yield (inputs,)
            else:
                yield (inputs, labels_batch)
