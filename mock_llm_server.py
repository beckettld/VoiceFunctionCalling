from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import openai
import ast
import os

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai.ChatCompletion

# Load the Gorilla LLM model
llm = Llama(model_path="./gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf", n_threads=8, n_gpu_layers=35)

# List of functions the model should recognize
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "control_lights",
        "description": "Control the lights",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["turn_on", "turn_off"],
                },
                "location": {
                    "type": "string",
                    "description": "The location of the lights, e.g., living room",
                },
            },
            "required": ["action", "location"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a timer",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "string",
                    "description": "The duration of the timer, e.g., 10 minutes",
                },
            },
            "required": ["duration"],
        },
    },
    {
        "name": "play_music",
        "description": "Play music",
        "parameters": {
            "type": "object",
            "properties": {
                "song_name": {
                    "type": "string",
                    "description": "The name of the song to play",
                },
                "artist": {
                    "type": "string",
                    "description": "The artist of the song",
                },
            },
            "required": ["song_name"],
        },
    },
    {
        "name": "find_recipe",
        "description": "Find a recipe",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "The main ingredient of the recipe",
                },
                "dietary_preference": {
                    "type": "string",
                    "enum": ["vegan", "vegetarian", "gluten-free", "none"],
                    "description": "Dietary preference for the recipe",
                },
            },
            "required": ["ingredient"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_city": {
                    "type": "string",
                    "description": "The city you are departing from",
                },
                "destination_city": {
                    "type": "string",
                    "description": "The city you are traveling to",
                },
                "date": {
                    "type": "string",
                    "description": "The date of the flight",
                },
            },
            "required": ["departure_city", "destination_city", "date"],
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting",
        "parameters": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "The person you want to meet with",
                },
                "time": {
                    "type": "string",
                    "description": "The time of the meeting",
                },
                "date": {
                    "type": "string",
                    "description": "The date of the meeting",
                },
            },
            "required": ["person", "time", "date"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to be reminded of",
                },
                "time": {
                    "type": "string",
                    "description": "The time of the reminder",
                },
                "date": {
                    "type": "string",
                    "description": "The date of the reminder",
                },
            },
            "required": ["task", "time", "date"],
        },
    },
    {
        "name": "order_food",
        "description": "Order food",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant": {
                    "type": "string",
                    "description": "The name of the restaurant",
                },
                "dish": {
                    "type": "string",
                    "description": "The name of the dish to order",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of dishes to order",
                },
            },
            "required": ["restaurant", "dish", "quantity"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "The email address of the recipient",
                },
                "subject": {
                    "type": "string",
                    "description": "The subject of the email",
                },
                "body": {
                    "type": "string",
                    "description": "The body of the email",
                },
            },
            "required": ["recipient", "subject", "body"],
        },
    }
]

def get_prompt(user_query: str, functions: list = []) -> str:
    system = "You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    if len(functions) == 0:
        return f"{system}\n### Instruction: <<question>> {user_query}\n### Response: "
    functions_string = json.dumps(functions)
    return f"{system}\n### Instruction: <<function>>{functions_string}\n<<question>>{user_query}\n### Response: "

def is_command(text: str) -> bool:
    prompt = f"Determine if the following text is a command that is associated with one of the defined functions in {functions}: '{text}'\nResponse with 'yes' or 'no':"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that determines if a given text should prompt an action that is associated with one of the given functions {functions}. Remember, even if the text is a command but it is not associated with any of the functions provided, still label it as \"not command\"."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )

    response_text = response.choices[0].message['content'].strip().lower()
    return "yes" in response_text

@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.json['text']
    
    # Log received command for debugging
    print(f"Text received: {text}")
    
    if is_command(text):
        print('***IS COMMAND***')

        def get_prompt(user_query: str, functions: list = []) -> str:
            system = "You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
            if len(functions) == 0:
                return f"{system}\n### Instruction: <<question>> {user_query}\n### Response: "
            functions_string = json.dumps(functions)
            return f"{system}\n### Instruction: <<function>>{functions_string}\n<<question>> {user_query}\n### Response: "

        def strip_function_calls(content: str) -> list[str]:
            return re.split(r'\n### Response:\n', content)

        def process_ast_node(node):
            # Check if the node is a function call
            if isinstance(node, ast.Call):
                # Return a string representation of the function call
                return ast.unparse(node)
            else:
                # Convert the node to source code and evaluate to get the value
                node_str = ast.unparse(node)
                return eval(node_str)

        def parse_python_function_call(call_str):
            tree = ast.parse(call_str)
            expr = tree.body[0]

            call_node = expr.value
            function_name = (
                call_node.func.id
                if isinstance(call_node.func, ast.Name)
                else str(call_node.func)
            )

            parameters = {}
            noNameParam = []

            # Process positional arguments
            for arg in call_node.args:
                noNameParam.append(process_ast_node(arg))

            # Process keyword arguments
            for kw in call_node.keywords:
                parameters[kw.arg] = process_ast_node(kw.value)

            if noNameParam:
                parameters["None"] = noNameParam

            function_dict = {"name": function_name, "arguments": parameters}
            return function_dict

        def parse_function_call(call: str) -> dict[str, any]:
            """
            This is temporary. The long term solution is to union all the
            types of the parameters from the user's input function definition,
            and check which language is a proper super set of the union type.
            """
            try:
                return parse_python_function_call(call)
            except Exception as e:
                return None

        def format_response(response: str):
            function_call_dicts = None
            try:
                response_list = strip_function_calls(response)
                if len(response_list) > 1:
                    function_call_dicts = []
                    for function_call in response_list:
                        function_call_dicts.append(parse_function_call(function_call))
                    response = ", ".join(response_list)
                else:
                    function_call_dicts = parse_function_call(response_list[0])
                    response = response_list[0]
            except Exception as e:
                pass
            return response, function_call_dicts

        prompt = get_prompt(text, functions=functions)

        # Generate response using the Llama model
        response = llm(prompt, max_tokens=128)

        # Process the generated response
        generated_text = response['choices'][0]['text']
        fn_call_string, function_call_dict = format_response(generated_text)
        print(fn_call_string)

    else:
        print('***NOT COMMAND***\n')
        response = {"function": "not_command", "parameters": {"text": text}}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)