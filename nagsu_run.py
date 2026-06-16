import sys
import json
import re

def load_nbf_bytecode(filepath):
    """
    Reads the Nagsu JSON file structure, validates the internal 
    magic_header field, and unpacks the memory registers.
    """
    print(f"[Nagsu Runtime] Initializing deployment stream from: {filepath}")
    try:
        with open(filepath, "r") as f:
            # Load directly as a JSON dictionary since it starts with {
            state_registry = json.load(f)
        
        # Verify the internal key from your file layout
        if state_registry.get("magic_header") != "NAGSU_CORE_BYTECODE":
            raise ValueError("Invalid Binary Error: Internal 'magic_header' verification failed.")
            
        return state_registry
        
    except FileNotFoundError:
        print(f"[Error] The target compiled file '{filepath}' could not be found.")
        sys.exit(1)
    except Exception as e:
        print(f"[Runtime Crash] Failed to unpack bytecode: {str(e)}")
        sys.exit(1)
def execute_matrix_multiplication(matrix_A, matrix_B):
    """Pure mathematical matrix multiplication (A x B) runner."""
    rows_A, cols_A = len(matrix_A), len(matrix_A[0])
    rows_B, cols_B = len(matrix_B), len(matrix_B[0])
    
    if cols_A != rows_B:
        raise ValueError(f"Shape Mismatch Runtime Error: Cannot multiply shapes [{rows_A}x{cols_A}] and [{rows_B}x{cols_B}]")
        
    # Pre-allocate zero-matrix for performance
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += matrix_A[i][k] * matrix_B[k][j]
    return result

def run_inference(bytecode, user_input_matrix):
    """Runs the optimized forward pass layers without compiling text code."""
    print("[Nagsu Runtime] Memory allocation successful. Executing layer operations...")
    
    # Target the 'tensors' dictionary from your exact file layout
    tensor_registry = bytecode.get("tensors", {})
    
    # Pull your compiled weights (W) and bias/input factors
    weights = tensor_registry.get("W", None)
    bias = tensor_registry.get("X", None) # Fallback vector offset
    
    if not weights:
        raise ValueError("Bytecode Corruption: Required model weights ('W') layer missing.")
        
    print(f" -> Input Layer Shape: [{len(user_input_matrix)}x{len(user_input_matrix[0])}]")
    print(f" -> Weights Registry Layer Shape: [{len(weights)}x{len(weights[0])}]")
    
    # Execute the primary core matrix transformation
    raw_output = execute_matrix_multiplication(user_input_matrix, weights)
    
    # Apply bias mapping if an offset vector exists
    if bias and isinstance(bias, list) and isinstance(bias[0], list):
        for i in range(min(len(raw_output), len(bias))):
            for j in range(min(len(raw_output[0]), len(bias[0]))):
                raw_output[i][j] += bias[i][j]
            
    print("[Nagsu Runtime] Inference Pass Complete!")
    return raw_output

if __name__ == "__main__":
    # Setup our standardized Command Line Interface (CLI) flags
    # Usage: python nagsu_run.py model.nbf "[[1, 2, 3, 4]]"
    if len(sys.argv) < 3:
        print("\n[Usage Error] Missing arguments!")
        print("Run command like this: python nagsu_run.py <compiled_file.nbf> \"<input_matrix_json>\"")
        print("Example: python nagsu_run.py model.nbf \"[[1.0, 2.0, 3.0, 4.0]]\"\n")
        sys.exit(1)
        
    model_file = sys.argv[1]
    raw_input_string = sys.argv[2]
    
    try:
        # Parse the input matrix argument string securely
        user_input = json.loads(raw_input_string)
        
        # Load the independent bytecode asset format
        bytecode_data = load_nbf_bytecode(model_file)
        
        # Stream the model prediction pass
        prediction_result = run_inference(bytecode_data, user_input)
        
        print("\n================ OUTPUT PREDICTION ================")
        print(prediction_result)
        print("===================================================\n")
        
    except json.JSONDecodeError:
        print("[Input Error] Provided input matrix must be valid JSON format. Example: \"[[1,2,3,4]]\"")
    except Exception as e:
        print(f"[Runtime Abort] System failed during runtime pass: {str(e)}")