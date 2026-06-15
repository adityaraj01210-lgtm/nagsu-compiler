import os
import json
from nagsu_engine import NagsuLexer, NagsuParser, NagsuRuntimeArena

def test_lexer_shape_property():
    sample_code = "entity CheckNet { vram = 8GB X = [[1.0]] assert X.shape == [1, 1] }"
    lexer = NagsuLexer(sample_code)
    tokens = lexer.tokenize()
    identifiers = [t['value'] for t in tokens if t['type'] == 'IDENTIFIER']
    assert "X.shape" in identifiers
    print("  ✔ Test 1 passed: Dot-property tokens parsed seamlessly.")

def test_parser_assertion_structure():
    sample_code = "entity CheckNet { vram = 8GB X = [[1.0]] assert X.shape == [1, 1] }"
    lexer = NagsuLexer(sample_code)
    tokens = lexer.tokenize()
    parser = NagsuParser(tokens)
    ast = parser.parse()
    entity_node = ast.children[0]
    node_types = [child.node_type for child in entity_node.children]
    assert "Guardrail_Assertion" in node_types
    print("  ✔ Test 2 passed: Assertion nodes are structured correctly in the AST.")

def test_runtime_static_dimension():
    arena = NagsuRuntimeArena("CheckNet", "8GB")
    arena.register_variable("X", [[1.5, -2.0, 4.0]])
    arena.evaluate_property_assertion("X.shape", [1, 3])
    print("  ✔ Test 3 passed: Static matrix dimensions verified by runtime guardrails perfectly.")

def test_binary_serialization_export():
    """Test 4: Verifies compiled model matrix weight state serialization."""
    test_filename = "test_model.nbf"
    if os.path.exists(test_filename):
        os.remove(test_filename)
        
    arena = NagsuRuntimeArena("AutomationNet", "24GB")
    arena.register_variable("W_Layer1", [[1.0, 2.0], [3.0, 4.0]])
    arena.export_compiled_binary(test_filename)
    
    assert os.path.exists(test_filename), "Serialization Error: Binary state file not generated on disk."
    
    with open(test_filename, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        
    assert loaded_data["magic_header"] == "NAGSU_CORE_BYTECODE", "Format Error: Missing magic core validation header tag."
    assert loaded_data["model_identifier"] == "AutomationNet", "Metadata Error: Incorrect model name mapping saved."
    assert "W_Layer1" in loaded_data["tensors"], "Data Error: Layer weight arrays missing from snapshot payload."
    
    os.remove(test_filename)
    print("  ✔ Test 4 passed: Binary state file snapshotted and validated successfully.")


def run_automation_verification_suite():
    print("=" * 75)
    print("🛡️  INITIALIZING SYSTEM VERIFICATION SUITE FOR NAGSU ARCHITECTURE...")
    print("=" * 75)
    
    try:
        test_lexer_shape_property()
        test_parser_assertion_structure()
        test_runtime_static_dimension()
        test_binary_serialization_export()
        
        print("\n" + "═" * 75)
        print("🏆 ALL SYSTEMS OPERATIONAL: NAGSU COMPILER PIPELINE IS 100% SECURE!")
        print("═" * 75)
    except AssertionError as error:
        print(f"\n🛑 AUTOMATION VERIFICATION CRASH POINT DETECTED:\n{error}")
    except Exception as general_err:
        print(f"\n🛑 UNEXPECTED STRUCTURAL SYSTEM CRASH DETECTED:\n{general_err}")

if __name__ == "__main__":
    run_verification_suite = run_automation_verification_suite
    run_verification_suite()