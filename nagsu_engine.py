import re
import os
import json
import ast as python_ast

# =====================================================================
# STAGE 1: UNIFIED PRODUCTION LEXER
# =====================================================================
class NagsuLexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        
    def tokenize(self):
        token_specification = [
            ('KEYWORD',    r'\b(entity|vram|compute|assert)\b'),
            ('COMPARE',    r'=='),
            ('ASSIGN',     r'='),
            ('OP',         r'[\*+]'),        
            ('ARRAY',      r'\[\s*\[.*?\]\s*\]|\[[\d\s,]+\]'), 
            ('IDENTIFIER', r'\b[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?\b'), 
            ('VALUE',      r'\b\d+[a-zA-Z]+\b|\b\d+\b'),
            ('LBRACE',     r'\{'),
            ('RBRACE',     r'\}'),
            ('SKIP',       r'[ \t\n\r]+'), 
            ('MISMATCH',   r'.'),            
        ]
        
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        
        for mo in re.finditer(tok_regex, self.source_code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(f"Lexer Error: Unexpected character: {value}")
            else:
                self.tokens.append({'type': kind, 'value': value})
        return self.tokens

# =====================================================================
# STAGE 2: ADAPTIVE PRODUCTION PARSER
# =====================================================================
class ASTNode:
    def __init__(self, node_type, value=None):
        self.node_type = node_type
        self.value = value
        self.children = []

    def __repr__(self, level=0):
        indent = "  " * level
        result = f"{indent}🚀 {self.node_type}: {self.value}\n"
        for child in self.children:
            result += child.__repr__(level + 1)
        return result

class NagsuParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def peek(self):
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return None

    def consume(self, expected_type, expected_value=None):
        token = self.peek()
        if not token:
            raise SyntaxError(f"Parser Error: Unexpected End of File.")
        if token['type'] != expected_type:
            raise SyntaxError(f"Parser Error: Expected '{expected_type}', got '{token['type']}'")
        if expected_value and token['value'] != expected_value:
            raise SyntaxError(f"Parser Error: Expected '{expected_value}', got '{token['value']}'")
        self.current += 1
        return token

    def parse(self):
        root = ASTNode("Program_Root")
        self.consume("KEYWORD", "entity")
        name_tok = self.consume("IDENTIFIER")
        entity_node = ASTNode("Model_Entity", name_tok['value'])
        
        self.consume("LBRACE")
        
        variables_node = ASTNode("Variable_Declarations")
        instruction_node = None
        assertion_node = None
        
        while self.peek() and self.peek()['type'] != 'RBRACE':
            token = self.peek()
            if token['type'] == 'KEYWORD' and token['value'] == 'vram':
                self.consume("KEYWORD", "vram")
                self.consume("ASSIGN")
                vram_val = self.consume("VALUE")
                entity_node.children.append(ASTNode("Hardware_Footprint_Target", vram_val['value']))
            elif token['type'] == 'KEYWORD' and token['value'] == 'compute':
                self.consume("KEYWORD", "compute")
                self.consume("ASSIGN")
                var1 = self.consume("IDENTIFIER")
                op1 = self.consume("OP")
                var2 = self.consume("IDENTIFIER")
                op2 = self.consume("OP")
                var3 = self.consume("IDENTIFIER")
                
                instruction_node = ASTNode("Chained_Compute_Instruction", "Layer_Forward")
                instruction_node.children.append(ASTNode("MatMul_Left", var1['value']))
                instruction_node.children.append(ASTNode("MatMul_Right", var2['value']))
                instruction_node.children.append(ASTNode("Op_Type", op1['value']))
                instruction_node.children.append(ASTNode("Bias_Term", var3['value']))
                instruction_node.children.append(ASTNode("Bias_Op", op2['value']))
            elif token['type'] == 'KEYWORD' and token['value'] == 'assert':
                self.consume("KEYWORD", "assert")
                
                if self.peek()['type'] == 'KEYWORD':
                    left_operand = self.consume("KEYWORD")
                else:
                    left_operand = self.consume("IDENTIFIER")
                    
                comp_op = self.consume("COMPARE")
                
                if self.peek()['type'] == 'ARRAY':
                    right_operand = self.consume("ARRAY")
                    parsed_right_val = python_ast.literal_eval(right_operand['value'])
                else:
                    right_operand = self.consume("VALUE")
                    parsed_right_val = right_operand['value']
                
                assertion_node = ASTNode("Guardrail_Assertion", comp_op['value'])
                assertion_node.children.append(ASTNode("Assert_Left", left_operand['value']))
                assertion_node.children.append(ASTNode("Assert_Right", parsed_right_val))
            elif token['type'] == 'IDENTIFIER':
                var_name = self.consume("IDENTIFIER")
                self.consume("ASSIGN")
                var_val = self.consume("ARRAY")
                parsed_array = python_ast.literal_eval(var_val['value'])
                variables_node.children.append(ASTNode(f"Variable:{var_name['value']}", parsed_array))
            else:
                self.current += 1
                
        self.consume("RBRACE")
        entity_node.children.append(variables_node)
        if instruction_node:
            entity_node.children.append(instruction_node)
        if assertion_node:
            entity_node.children.append(assertion_node)
            
        root.children.append(entity_node)
        return root

# =====================================================================
# STAGE 3: RUNTIME ENGINE WITH MODEL EXPORT SERIALIZER
# =====================================================================
class NagsuRuntimeArena:
    def __init__(self, entity_name, memory_target):
        self.entity_name = entity_name
        self.memory_target = memory_target
        self.variable_registry = {}

    def register_variable(self, name, value):
        self.variable_registry[name] = value

    def get_variable(self, name):
        if name not in self.variable_registry:
            raise NameError(f"Nagsu Runtime Error: Variable '{name}' is undefined.")
        return self.variable_registry[name]

    def execute_matmul(self, matrix_a, matrix_b):
        rows_a, cols_a = len(matrix_a), len(matrix_a[0])
        rows_b, cols_b = len(matrix_b), len(matrix_b[0])
        if cols_a != rows_b:
            raise ValueError(f"Nagsu Runtime Error: Dimension Mismatch ({rows_a}x{cols_a}) vs ({rows_b}x{cols_b}).")
        
        output_matrix = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                dot_product = 0.0
                for k in range(cols_a):
                    dot_product += matrix_a[i][k] * matrix_b[k][j]
                output_matrix[i][j] = dot_product
        return output_matrix

    def execute_elementwise_add(self, matrix_a, matrix_b):
        rows_a, cols_a = len(matrix_a), len(matrix_a[0])
        rows_b, cols_b = len(matrix_b), len(matrix_b[0])
        if rows_a != rows_b or cols_a != cols_b:
            raise ValueError(f"Nagsu Runtime Error: Bias addition shape mismatch ({rows_a}x{cols_a}) vs ({rows_b}x{cols_b}).")
            
        output_matrix = [[0.0 for _ in range(cols_a)] for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_a):
                output_matrix[i][j] = matrix_a[i][j] + matrix_b[i][j]
        return output_matrix

    def evaluate_property_assertion(self, left_id_string, expected_val):
        if ".shape" in left_id_string:
            target_var_name = left_id_string.split(".")[0]
            matrix_data = self.get_variable(target_var_name)
            actual_shape = [len(matrix_data), len(matrix_data[0])]
            if list(actual_shape) != list(expected_val):
                raise AssertionError(f"Nagsu Shape Error: Matrix '{target_var_name}' size constraint failed! Expected {expected_val}, but evaluated to {actual_shape}.")
            print(f"  ✔ [Guardrail Passed] Dimension check: '{target_var_name}.shape' matches constraints {expected_val}.")
        elif left_id_string == "vram":
            if self.memory_target != expected_val:
                raise AssertionError(f"Nagsu Environmental Violation: Expected vram to be {expected_val}, got {self.memory_target}.")
            print(f"  ✔ [Guardrail Passed] Assertion match: 'vram' successfully verified as {expected_val}.")

    def compute_native_relu(self, input_matrix):
        return [[max(0.0, float(element)) for element in row] for row in input_matrix]

    def export_compiled_binary(self, filename="model.nbf"):
        """Serializes current memory state into Nagsu Bytecode Format."""
        compiled_payload = {
            "magic_header": "NAGSU_CORE_BYTECODE",
            "model_identifier": self.entity_name,
            "target_hardware_vram": self.memory_target,
            "tensors": self.variable_registry
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(compiled_payload, f, indent=4)
        print(f"  ✔ [Serialization Success] Snapshot compiled and written to disk -> '{filename}'")

    def shutdown(self):
        print("\n*** [Runtime Clean-up] Structural runtime execution contexts successfully freed.")


def run_system_pipeline():
    target_source_file = "main.nsg"
    if not os.path.exists(target_source_file):
        raise FileNotFoundError(f"Compiler System Error: Source script missing.")
    with open(target_source_file, "r", encoding="utf-8") as f:
        nagsu_source_code = f.read()
    
    print("⚡ STAGE 1 & 2: TRANSLATING DIMENSION PROPERTIES...")
    lexer = NagsuLexer(nagsu_source_code)
    tokens = lexer.tokenize()
    parser = NagsuParser(tokens)
    ast = parser.parse()
    print(ast)
    
    entity_node = ast.children[0]
    entity_name = entity_node.value
    vram_size = "Unknown"
    variables_list = []
    instruction_node = None
    assertion_node = None
    
    for child in entity_node.children:
        if child.node_type == "Hardware_Footprint_Target":
            vram_size = child.value
        elif child.node_type == "Variable_Declarations":
            variables_list = child.children
        elif child.node_type == "Chained_Compute_Instruction":
            instruction_node = child
        elif child.node_type == "Guardrail_Assertion":
            assertion_node = child
            
    print("🚀 STAGE 3: RUNNING TENSOR SYMBOL VALIDATION PASS...")
    arena = NagsuRuntimeArena(entity_name, vram_size)
    for var_node in variables_list:
        var_name = var_node.node_type.split(":")[1]
        arena.register_variable(var_name, var_node.value)
        
    if assertion_node:
        left_operand = assertion_node.children[0].value
        right_operand = assertion_node.children[1].value
        arena.evaluate_property_assertion(left_operand, right_operand)
        
    if instruction_node:
        mat_left = arena.get_variable(instruction_node.children[0].value)
        mat_right = arena.get_variable(instruction_node.children[1].value)
        mat_bias = arena.get_variable(instruction_node.children[3].value)
        
        matmul_result = arena.execute_matmul(mat_left, mat_right)
        full_layer_result = arena.execute_elementwise_add(matmul_result, mat_bias)
        activated_output = arena.compute_native_relu(full_layer_result)
        
        # Register the final calculated array state to memory tracking before file dump
        arena.register_variable("activated_output", activated_output)
        
        formatted_matrix = ", ".join([f"{x:.2f}" for x in activated_output[0]])
        print(f"\n⚡ Output Matrix Layer Result:\n    [ {formatted_matrix} ]")
        
    print("\n📦 STAGE 4: TRIGGERING COMPILED WEIGHT EXPORT...")
    arena.export_compiled_binary("model.nbf")
    
    print("-" * 75)
    arena.shutdown()

if __name__ == "__main__":
    run_system_pipeline()