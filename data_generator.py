import h5py
import numpy as np

try:
    from tensorflow.keras.utils import Sequence
except ImportError:
    from keras.utils import Sequence


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


class H5DataGenerator(Sequence):
    """Sequence generator that yields batches of data from H5 file with low memory profile."""
    
    def __init__(self, config, data_type, mixture_num=None):
        self.config = config
        self.data_type = data_type
        self.mixture_num = mixture_num
        self.batch_size = config['batch_size']
        self.data_file = config.get('processed_h5')
        
        if not self.data_file:
            num_mon_sites = config['num_mon_sites']
            num_mon_inst_train = config['num_mon_inst_train']
            num_mon_inst_test = config['num_mon_inst_test']
            num_mon_inst = num_mon_inst_train + num_mon_inst_test
            num_unmon_sites_train = config['num_unmon_sites_train']
            num_unmon_sites_test = config['num_unmon_sites_test']
            data_dir = config['data_dir']
            self.data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites, num_mon_inst,
                                              num_unmon_sites_train, num_unmon_sites_test)
            
        # Determine sequence and metadata configuration
        if "sequence_dataset" in self.config:
            # New flat config style
            self.seq_ds_name = self.config["sequence_dataset"]
            self.seq_input_name = self.config["sequence_input_name"]
            self.meta_ds_name = self.config.get("metadata_dataset")
            self.meta_type = self.config.get("metadata_type")
            self.wfmeta_k = self.config.get("wfmeta_k", 10)
        else:
            # Backward compatibility fallback
            if self.mixture_num is None:
                raise ValueError("Either mixture_num must be provided or sequence_dataset must be in config.")
            mixture = self.config['mixture']
            inner_comb = mixture[self.mixture_num]
            
            self.seq_ds_name = None
            self.seq_input_name = None
            if 'dir' in inner_comb:
                self.seq_ds_name = 'dir_seq'
                self.seq_input_name = 'dir_input'
            elif 'time' in inner_comb:
                self.seq_ds_name = 'time_seq'
                self.seq_input_name = 'time_input'
            elif 'dir_iat_log' in inner_comb:
                self.seq_ds_name = 'dir_iat_log'
                self.seq_input_name = 'dir_iat_log_input'
            elif 'dir_x_iat' in inner_comb:
                self.seq_ds_name = 'dir_x_iat'
                self.seq_input_name = 'dir_x_iat_input'
            elif 'dir_iat_raw' in inner_comb:
                self.seq_ds_name = 'dir_iat_raw'
                self.seq_input_name = 'dir_iat_raw_input'
                
            self.meta_ds_name = 'metadata' if 'metadata' in inner_comb else None
            self.meta_type = 'metadata'
            self.wfmeta_k = 10
            
        # Open file once to verify total sample size
        with h5py.File(self.data_file, 'r') as f:
            self.num_samples = len(f[self.data_type]['labels'])
            
        self.indices = np.arange(self.num_samples)
        
        # Only shuffle training data
        self.shuffle = (self.data_type == 'training_data')
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Initialize file and dataset references as None for lazy evaluation (thread-safety)
        self.f = None
        self.grp = None
        self.seq_ds = None
        self.meta_ds = None
        self.labels_ds = None
        
    def _open_file(self):
        """Open H5 file and retrieve dataset links lazily."""
        if self.f is None:
            self.f = h5py.File(self.data_file, 'r')
            self.grp = self.f[self.data_type]
            self.seq_ds = self.grp[self.seq_ds_name] if (self.seq_ds_name and self.seq_ds_name in self.grp) else None
            self.meta_ds = self.grp[self.meta_ds_name] if (self.meta_ds_name and self.meta_ds_name in self.grp) else None
            self.labels_ds = self.grp['labels']

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        self._open_file()
        
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_samples)
        
        # Safety fallback
        if start >= self.num_samples:
            start = 0
            end = self.batch_size
            
        batch_indices = self.indices[start:end]
        
        inputs = {}
        
        # Load sequence batch
        if self.seq_ds is not None:
            seq_batch = safe_h5_read(self.seq_ds, batch_indices)
            inputs[self.seq_input_name] = seq_batch.astype(np.float32)

        # Load metadata batch
        if self.meta_ds is not None:
            meta_batch = safe_h5_read(self.meta_ds, batch_indices)
            if self.meta_type == "wfmeta10" or self.meta_ds_name == "wfmeta":
                meta_batch = meta_batch[:, :self.wfmeta_k]
            inputs['metadata_input'] = meta_batch.astype(np.float32)

        labels_batch = safe_h5_read(self.labels_ds, batch_indices).astype(np.float32)

        if self.data_type == 'test_data':
            return (inputs,)
        else:
            return (inputs, labels_batch)

    def on_epoch_end(self):
        """Re-shuffles training indices at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def close(self):
        """Close the H5 file handle if open."""
        if self.f is not None:
            self.f.close()
            self.f = None
            self.grp = None
            self.seq_ds = None
            self.meta_ds = None
            self.labels_ds = None


def generate(config, data_type, mixture_num=None):
    """Wrapper function returning Sequence generator instance."""
    return H5DataGenerator(config, data_type, mixture_num)
