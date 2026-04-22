import os
import random
import json

def get_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def generate_dummy_file(filepath):
    # Generates a sequence of 50 to 100 packets
    num_packets = random.randint(50, 100)
    lines = []
    current_time = 0.0
    for _ in range(num_packets):
        # time increment
        current_time += random.uniform(0.001, 0.05)
        # direction 1 or -1
        direction = 1 if random.random() < 0.5 else -1
        # size around 500-1500 doesn't matter much as only sign(dir) is used in wang_to_varcnn.py line 31
        dir_val = direction * random.randint(500, 1500)
        lines.append(f"{current_time:.4f}\t{dir_val}\n")
    
    with open(filepath, 'w') as f:
        f.writelines(lines)

def main():
    config = get_config()
    num_mon_sites = config['num_mon_sites']
    num_mon_inst = config['num_mon_inst_train'] + config['num_mon_inst_test']
    num_unmon = config['num_unmon_sites_train'] + config['num_unmon_sites_test']

    output_dir = os.path.join(config.get('data_dir', 'data/'), 'batch_wang')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating dummy data in {output_dir}")
    print(f"Monitored sites: {num_mon_sites}, Instances per site: {num_mon_inst}")
    print(f"Unmonitored sites: {num_unmon}")

    count: int = 0
    # Generate monitored instances
    for site in range(num_mon_sites):
        for inst in range(num_mon_inst):
            filename = f"{site}-{inst}"
            filepath = os.path.join(output_dir, filename)
            generate_dummy_file(filepath)
            count += 1
            if count % 100 == 0:
                print(f"Generated {count} files")

    # Generate unmonitored instances
    for inst in range(num_unmon):
        filename = f"{inst}"
        filepath = os.path.join(output_dir, filename)
        generate_dummy_file(filepath)
        count += 1  # type: ignore
        if count % 100 == 0:
            print(f"Generated {count} files")
            
    print(f"Finished generating {count} dummy files.")

if __name__ == '__main__':
    main()
