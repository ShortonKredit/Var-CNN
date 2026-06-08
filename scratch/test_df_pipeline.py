import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import df

def test_config_build(config_path, expected_classes):
    print(f"Testing config: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path {config_path} does not exist!")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize model
    model, callbacks = df.get_model(config)
    
    # Check input
    input_layer = model.inputs[0]
    expected_input_name = config.get("sequence_input_name", "dir_input")
    actual_input_name = input_layer.name.split(":")[0]  # strip Keras suffix if any
    
    print(f"  Input layer name: {input_layer.name} (Expected key suffix: {expected_input_name})")
    print(f"  Input layer shape: {input_layer.shape}")
    
    # Ensure expected_input_name is a substring of actual_input_name (due to Keras tensor naming)
    assert expected_input_name in actual_input_name, f"Mismatch input name: expected {expected_input_name} in {actual_input_name}"
    assert input_layer.shape[1] == config["seq_length"], f"Mismatch seq_length: {input_layer.shape[1]} != {config['seq_length']}"
    
    # Check output classes
    output_layer = model.outputs[0]
    actual_classes = output_layer.shape[-1]
    print(f"  Output layer shape: {output_layer.shape} (Expected classes: {expected_classes})")
    assert actual_classes == expected_classes, f"Mismatch classes: {actual_classes} != {expected_classes}"
    
    # Check callbacks
    print(f"  Number of callbacks generated: {len(callbacks)}")
    assert len(callbacks) > 0, "No callbacks generated!"
    
    # Mock inference to verify compilation and layer math
    mock_input = np.random.randint(0, 2, size=(2, config["seq_length"], 1)).astype(np.float32)
    input_dict = {expected_input_name: mock_input}
    predictions = model.predict(input_dict, verbose=0)
    print(f"  Mock inference output shape: {predictions.shape}")
    assert predictions.shape == (2, expected_classes), f"Mismatch predictions shape: {predictions.shape}"
    print(f"  [PASS] Config {config_path} built and verified successfully!\n")

def main():
    test_config_build("configs/df/df_cw_dir.json", expected_classes=100)
    test_config_build("configs/df/df_ow_dir.json", expected_classes=101)
    print("All DF models built and verified successfully!")

if __name__ == "__main__":
    main()
