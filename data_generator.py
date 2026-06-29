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


def generate(config, data_type):
    """Yields batch of data with the correct content and formatting.

    Args:
        config (dict): Deserialized JSON config file (see config.json)
        data_type (str): Either 'training_data', 'validation_data', or
            'test_data'
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
    seq_ds_name = config["sequence_dataset"]
    seq_input_name = config["sequence_input_name"]
    meta_ds_name = config.get("metadata_dataset")
    meta_type = config.get("metadata_type")
    wfmeta_k = config.get("wfmeta_k", 10)

    # Stream data from H5 file batch by batch
    with h5py.File(data_file, 'r') as f:
        grp = f[data_type]
        
        seq_ds = grp[seq_ds_name] if (seq_ds_name and seq_ds_name in grp) else None
        meta_ds = grp[meta_ds_name] if (meta_ds_name and meta_ds_name in grp) else None
        labels_ds = grp['labels']
        
        if data_type == 'training_data' and config.get("train_indices_file"):
            indices_path = config["train_indices_file"]
            sample_indices = np.load(indices_path)
            num_samples = len(sample_indices)
            order = np.arange(num_samples)
            np.random.shuffle(order)
        else:
            num_samples = len(labels_ds)
            sample_indices = np.arange(num_samples)
            order = np.arange(num_samples)
            if data_type == 'training_data':
                np.random.shuffle(order)

        batch_start = 0
        while True:
            if batch_start >= num_samples:
                batch_start = 0
                if data_type == 'training_data':
                    np.random.shuffle(order)

            batch_positions = order[batch_start:batch_start + batch_size]
            batch_start += batch_size

            batch_indices = sample_indices[batch_positions]

            inputs = {}
            
            # Load sequence batch
            if seq_ds is not None:
                seq_batch = safe_h5_read(seq_ds, batch_indices)
                inputs[seq_input_name] = seq_batch.astype(np.float32)

            # Load metadata batch
            if meta_ds is not None:
                meta_batch = safe_h5_read(meta_ds, batch_indices)
                if (meta_type and meta_type.startswith("wfmeta")) or meta_ds_name == "wfmeta":
                    # Extract top-k ANOVA features (defaults to 10)
                    meta_batch = meta_batch[:, :wfmeta_k]
                inputs['metadata_input'] = meta_batch.astype(np.float32)

            labels_batch = safe_h5_read(labels_ds, batch_indices).astype(np.float32)
            if config.get("scenario") == "open_world":
                num_mon_sites = config["num_mon_sites"]
                class_indices = np.argmax(labels_batch, axis=1)
                binary_labels = np.zeros((len(labels_batch), 2), dtype=np.float32)
                binary_labels[class_indices < num_mon_sites, 0] = 1.0
                binary_labels[class_indices == num_mon_sites, 1] = 1.0
                labels_batch = binary_labels

            # Test data does not yield labels
            if data_type == 'test_data':
                yield (inputs,)
            else:
                yield (inputs, labels_batch)
