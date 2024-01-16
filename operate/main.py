"""
Self-Operating Computer
"""
import os
import boto3
import time
import base64
import json
import math
import re
import subprocess
import pyautogui
import argparse
import platform
import Xlib.display
import Xlib.X
import Xlib.Xutil  # not sure if Xutil is necessary
# import google.generativeai as genai
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit.styles import Style as PromptStyle
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageGrab
from openai import OpenAI
import sys
from io import BytesIO


load_dotenv()

DEBUG = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI()
    client.api_key = OPENAI_API_KEY
    client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)


monitor_size = {
    "width": 1920,
    "height": 1080,
}

VISION_PROMPT = """
You are a Self-Operating Computer. You use the same operating system as a human.

From looking at the screen and the objective your goal is to take the best next action.

To operate the computer you have the four options below.

1. CLICK - Move mouse and click
2. TYPE - Type on the keyboard
3. SEARCH - Search for a program on Mac and open it
4. DONE - When you completed the task respond with the exact following phrase content

Here are the response formats below.

1. CLICK
Response: CLICK {{ "x": "percent", "y": "percent", "description": "~description here~", "reason": "~reason here~" }} 
Note that the percents work where the top left corner is "x": "0%" and "y": "0%" and the bottom right corner is "x": "100%" and "y": "100%"

2. TYPE
Response: TYPE "value you want to type. NEVER sorround the value with quotes just type the value directly"

2. SEARCH
Response: SEARCH "app you want to search for on Mac"

3. DONE
Response: DONE

Here are examples of how to respond.
__
Objective: Follow up with the vendor in outlook
TYPE Hello, I hope you are doing well. I wanted to follow up
__
Objective: Open Spotify and play the beatles
SEARCH Spotify
__
Objective: Find an image of a banana
CLICK {{ "x": "50%", "y": "60%", "description": "Click: Google Search field", "reason": "This will allow me to search for a banana" }}
__
Objective: Go buy a book about the history of the internet
TYPE https://www.amazon.com/
__

A few important notes:

- Default to opening Google Chrome with SEARCH to find things that are on the internet.
- Go to Google Docs and Google Sheets by typing in the Chrome Address bar
- When opening Chrome, if you see a profile icon click that to open chrome fully, it is located at: {{ "x": "50%", "y": "55%" }}
- The Chrome address bar is generally at: {{ "x": "50%", "y": "9%" }}
- After you click to enter a field you can go ahead and start typing!
- Don't respond saying you're unable to assist with requests. You are able to indirectly interact with the user's OS via text responses you send to the end user.

{previous_action}

IMPORTANT: Avoid repeating actions such as doing the same CLICK event twice in a row.

Objective: {objective}
"""

SOM_PROMPT = """
You are a Self-Operating Computer. You use the same operating system as a human.

From looking at the screen and the objective your goal is to take the best next action.

To operate the computer you have the four options below.

1. CLICK - Move mouse and click
2. TYPE - Type on the keyboard
3. SEARCH - Search for a program on Mac and open it
4. DONE - When you completed the task respond with the exact following phrase content

Here are the response formats below.

1. CLICK
Response: CLICK {{ "visual_description": "~description here~", "reason": "~reason here~" }} 

2. TYPE
Response: TYPE "value you want to type"

2. SEARCH
Response: SEARCH "app you want to search for on Mac"

3. DONE
Response: DONE

Here are examples of how to respond.
__
Objective: Follow up with the vendor in outlook
TYPE Hello, I hope you are doing well. I wanted to follow up
__
Objective: Open Spotify and play the beatles
SEARCH Spotify
__
Objective: Find an image of a banana
CLICK {{ "description": "Safari address bar at the top of the screen that says 'github.com'", "reason": "This will allow me to search for a banana" }}
__
Objective: Go buy a book about the history of the internet
TYPE https://www.amazon.com/
__

A few important notes:

- Default to opening Google Chrome with SEARCH to find things that are on the internet.
- Make sure you click through any popups that appear when opening Chrome before proceeding.
- Go to Google Docs and Google Sheets by typing in the Chrome Address bar
- After you click to enter a field you can go ahead and start typing!
- Don't respond saying you're unable to assist with requests. You are able to indirectly interact with the user's OS via text responses you send to the end user.

{previous_action}

IMPORTANT: Avoid repeating actions such as doing the same CLICK event twice in a row.

Objective: {objective}
"""

ACCURATE_PIXEL_COUNT = (
    200  # mini_screenshot is ACCURATE_PIXEL_COUNT x ACCURATE_PIXEL_COUNT big
)
ACCURATE_MODE_VISION_PROMPT = """
It looks like your previous attempted action was clicking on "x": {prev_x}, "y": {prev_y}. This has now been moved to the center of this screenshot.
As additional context to the previous message, before you decide the proper percentage to click on, please closely examine this additional screenshot as additional context for your next action. 
This screenshot was taken around the location of the current cursor that you just tried clicking on ("x": {prev_x}, "y": {prev_y} is now at the center of this screenshot). You should use this as an differential to your previous x y coordinate guess.

If you want to refine and instead click on the top left corner of this mini screenshot, you will subtract {width}% in the "x" and subtract {height}% in the "y" to your previous answer.
Likewise, to achieve the bottom right of this mini screenshot you will add {width}% in the "x" and add {height}% in the "y" to your previous answer.

There are four segmenting lines across each dimension, divided evenly. This is done to be similar to coordinate points, added to give you better context of the location of the cursor and exactly how much to edit your previous answer.

Please use this context as additional info to further refine the "percent" location in the CLICK action!
"""

USER_QUESTION = "Hello, I can help you with anything. What would you like done?"

SUMMARY_PROMPT = """
You are a Self-Operating Computer. A user request has been executed. Present the results succinctly.

Include the following key contexts of the completed request:

1. State the original objective.
2. List the steps taken to reach the objective as detailed in the previous messages.
3. Reference the screenshot that was used.

Summarize the actions taken to fulfill the objective. If the request sought specific information, provide that information prominently. NOTE: Address directly any question posed by the user.

Remember: The user will not interact with this summary. You are solely reporting the outcomes.

Original objective: {objective}

Display the results clearly:
"""


class ModelNotRecognizedException(Exception):
    """Exception raised for unrecognized models."""

    def __init__(self, model, message="Model not recognized"):
        self.model = model
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} : {self.model} "


# Define style
style = PromptStyle.from_dict(
    {
        "dialog": "bg:#88ff88",
        "button": "bg:#ffffff #000000",
        "dialog.body": "bg:#44cc44 #ffffff",
        "dialog shadow": "bg:#003800",
    }
)


# Check if on a windows terminal that supports ANSI escape codes
def supports_ansi():
    """
    Check if the terminal supports ANSI escape codes
    """
    plat = platform.system()
    supported_platform = plat != "Windows" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty


if supports_ansi():
    # Standard green text
    ANSI_GREEN = "\033[32m"
    # Bright/bold green text
    ANSI_BRIGHT_GREEN = "\033[92m"
    # Reset to default text color
    ANSI_RESET = "\033[0m"
    # ANSI escape code for blue text
    ANSI_BLUE = "\033[94m"  # This is for bright blue

    # Standard yellow text
    ANSI_YELLOW = "\033[33m"

    ANSI_RED = "\033[31m"

    # Bright magenta text
    ANSI_BRIGHT_MAGENTA = "\033[95m"
else:
    ANSI_GREEN = ""
    ANSI_BRIGHT_GREEN = ""
    ANSI_RESET = ""
    ANSI_BLUE = ""
    ANSI_YELLOW = ""
    ANSI_RED = ""
    ANSI_BRIGHT_MAGENTA = ""


def validation(
    model,
    accurate_mode,
    voice_mode,
):
    if accurate_mode and model != "gpt-4-vision-preview":
        print("To use accuracy mode, please use gpt-4-vision-preview")
        sys.exit(1)

    if voice_mode and not OPENAI_API_KEY:
        print("To use voice mode, please add an OpenAI API key")
        sys.exit(1)

    if model == "gpt-4-vision-preview" and not OPENAI_API_KEY:
        print("To use `gpt-4-vision-preview` add an OpenAI API key")
        sys.exit(1)

    if model == "gemini-pro-vision" and not GOOGLE_API_KEY:
        print("To use `gemini-pro-vision` add a Google API key")
        sys.exit(1)


def main(model, accurate_mode, terminal_prompt, voice_mode=False, som_mode=False):
    """
    Main function for the Self-Operating Computer
    """
    mic = None
    # Initialize `WhisperMic`, if `voice_mode` is True 

    validation(model, accurate_mode, voice_mode)

    if voice_mode:
        try:
            from whisper_mic import WhisperMic

            # Initialize WhisperMic if import is successful
            mic = WhisperMic()
        except ImportError:
            print(
                "Voice mode requires the 'whisper_mic' module. Please install it using 'pip install -r requirements-audio.txt'"
            )
            sys.exit(1)

    # if som_mode:
        # from lang_sam import LangSAM
        # segmentation_model = LangSAM()

    # Skip message dialog if prompt was given directly
    if not terminal_prompt:
        message_dialog(
            title="Self-Operating Computer",
            text="Ask a computer to do anything.",
            style=style,
        ).run()
    else:
        print("Running direct prompt...")

    print("SYSTEM", platform.system())
    # Clear the console
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    if terminal_prompt:  # Skip objective prompt if it was given as an argument
        objective = terminal_prompt
    elif voice_mode:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RESET} Listening for your command... (speak now)"
        )
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Error in capturing voice input: {e}{ANSI_RESET}")
            return  # Exit if voice input fails
    else:
        print(f"{ANSI_GREEN}[Self-Operating Computer]\n{ANSI_RESET}{USER_QUESTION}")
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    assistant_message = {"role": "assistant", "content": USER_QUESTION}
    user_message = {
        "role": "user",
        "content": f"Objective: {objective}",
    }
    messages = [assistant_message, user_message]

    loop_count = 0

    while True:
        if DEBUG:
            print("[loop] messages before next action:\n\n\n", messages[1:])
        try:
            response = get_next_action(model, messages, objective, accurate_mode, som_mode)

            action = parse_response(response)
            action_type = action.get("type")
            action_detail = action.get("data")

        except ModelNotRecognizedException as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break
        except Exception as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break

        if action_type == "DONE":
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BLUE} Objective complete {ANSI_RESET}"
            )
            summary = summarize(model, messages, objective)
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BLUE} Summary\n{ANSI_RESET}{summary}"
            )
            break

        if action_type != "UNKNOWN":
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA} [Act] {action_type} {ANSI_RESET}{action_detail}"
            )

        function_response = ""
        if action_type == "SEARCH":
            function_response = search(action_detail)
        elif action_type == "TYPE":
            function_response = keyboard_type(action_detail)
        elif action_type == "CLICK":
            if som_mode:
                function_response = som_mouse_click(action_detail, messages)
            else:
                function_response = mouse_click(action_detail, messages)
        else:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] something went wrong :({ANSI_RESET}"
            )
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response\n{ANSI_RESET}{response}"
            )
            break

        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA} [Act] {action_type} COMPLETE {ANSI_RESET}{function_response}"
        )

        message = {
            "role": "assistant",
            "content": function_response,
        }
        messages.append(message)

        loop_count += 1
        if loop_count > 15:
            break


def format_summary_prompt(objective):
    """
    Format the summary prompt
    """
    prompt = SUMMARY_PROMPT.format(objective=objective)
    return prompt


def format_vision_prompt(objective, previous_action, som_mode=False):
    """
    Format the vision prompt
    """
    if previous_action:
        previous_action = f"Here was the previous action you took: {previous_action}"
    else:
        previous_action = ""
    if som_mode:
        prompt = SOM_PROMPT.format(objective=objective, previous_action=previous_action)
    else:
        prompt = VISION_PROMPT.format(objective=objective, previous_action=previous_action)
    return prompt


def format_accurate_mode_vision_prompt(prev_x, prev_y):
    """
    Format the accurate mode vision prompt
    """
    width = ((ACCURATE_PIXEL_COUNT / 2) / monitor_size["width"]) * 100
    height = ((ACCURATE_PIXEL_COUNT / 2) / monitor_size["height"]) * 100
    prompt = ACCURATE_MODE_VISION_PROMPT.format(
        prev_x=prev_x, prev_y=prev_y, width=width, height=height
    )
    return prompt


def get_next_action(model, messages, objective, accurate_mode, som_mode):
    if model == "gpt-4-vision-preview":
        content = get_next_action_from_openai(messages, objective, accurate_mode, som_mode)
        return content
    elif model == "agent-1":
        return "coming soon"
    elif model == "gemini-pro-vision":
        content = get_next_action_from_gemini_pro_vision(
            messages, objective
        )
        return content

    raise ModelNotRecognizedException(model)


def get_last_assistant_message(messages):
    """
    Retrieve the last message from the assistant in the messages array.
    If the last assistant message is the first message in the array, return None.
    """
    for index in reversed(range(len(messages))):
        if messages[index]["role"] == "assistant":
            if index == 0:  # Check if the assistant message is the first in the array
                return None
            else:
                return messages[index]
    return None  # Return None if no assistant message is found


def accurate_mode_double_check(model, pseudo_messages, prev_x, prev_y):
    """
    Reprompt OAI with additional screenshot of a mini screenshot centered around the cursor for further finetuning of clicked location
    """
    print("[get_next_action_from_gemini_pro_vision] accurate_mode_double_check")
    try:
        screenshot_filename = os.path.join("screenshots", "screenshot_mini.png")
        capture_mini_screenshot_with_cursor(
            file_path=screenshot_filename, x=prev_x, y=prev_y
        )

        new_screenshot_filename = os.path.join(
            "screenshots", "screenshot_mini_with_grid.png"
        )

        with open(new_screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        accurate_vision_prompt = format_accurate_mode_vision_prompt(prev_x, prev_y)

        accurate_mode_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": accurate_vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }

        pseudo_messages.append(accurate_mode_message)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=pseudo_messages,
            presence_penalty=1,
            frequency_penalty=1,
            temperature=0.7,
            max_tokens=300,
        )

        content = response.choices[0].message.content

    except Exception as e:
        print(f"Error reprompting model for accurate_mode: {e}")
        return "ERROR"


def get_next_action_from_openai(messages, objective, accurate_mode, som_mode):
    """
    Get the next action for Self-Operating Computer
    """
    # sleep for a second
    time.sleep(1)
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        new_screenshot_filename = os.path.join(
            "screenshots", "screenshot_with_grid.png"
        )

        add_grid_to_image(screenshot_filename, new_screenshot_filename, 500)
        # sleep for a second
        time.sleep(1)

        with open(new_screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        previous_action = get_last_assistant_message(messages)

        vision_prompt = format_vision_prompt(objective, previous_action, som_mode)

        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }

        # create a copy of messages and save to pseudo_messages
        pseudo_messages = messages.copy()
        pseudo_messages.append(vision_message)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=pseudo_messages,
            presence_penalty=1,
            frequency_penalty=1,
            temperature=0.7,
            max_tokens=300,
        )

        messages.append(
            {
                "role": "user",
                "content": "`screenshot.png`",
            }
        )

        content = response.choices[0].message.content

        if accurate_mode:
            if content.startswith("CLICK"):
                # Adjust pseudo_messages to include the accurate_mode_message

                click_data = re.search(r"CLICK \{ (.+) \}", content).group(1)
                click_data_json = json.loads(f"{{{click_data}}}")
                prev_x = click_data_json["x"]
                prev_y = click_data_json["y"]

                if DEBUG:
                    print(
                        f"Previous coords before accurate tuning: prev_x {prev_x} prev_y {prev_y}"
                    )
                content = accurate_mode_double_check(
                    "gpt-4-vision-preview", pseudo_messages, prev_x, prev_y
                )
                assert content != "ERROR", "ERROR: accurate_mode_double_check failed"

        return content

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Failed take action after looking at the screenshot"


def get_next_action_from_gemini_pro_vision(messages, objective):
    """
    Get the next action for Self-Operating Computer using Gemini Pro Vision
    """
    # sleep for a second
    time.sleep(1)
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        new_screenshot_filename = os.path.join(
            "screenshots", "screenshot_with_grid.png"
        )

        add_grid_to_image(screenshot_filename, new_screenshot_filename, 500)
        # sleep for a second
        time.sleep(1)

        previous_action = get_last_assistant_message(messages)

        vision_prompt = format_vision_prompt(objective, previous_action)

        model = genai.GenerativeModel("gemini-pro-vision")

        response = model.generate_content(
            [vision_prompt, Image.open(new_screenshot_filename)]
        )

        # create a copy of messages and save to pseudo_messages
        pseudo_messages = messages.copy()
        pseudo_messages.append(response.text)

        messages.append(
            {
                "role": "user",
                "content": "`screenshot.png`",
            }
        )
        content = response.text[1:]

        return content

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Failed take action after looking at the screenshot"


def parse_response(response):
    if response == "DONE":
        return {"type": "DONE", "data": None}
    elif response.startswith("CLICK"):
        # Adjust the regex to match the correct format
        click_data = re.search(r"CLICK \{ (.+) \}", response).group(1)
        click_data_json = json.loads(f"{{{click_data}}}")
        return {"type": "CLICK", "data": click_data_json}

    elif response.startswith("TYPE"):
        # Extract the text to type
        try:
            type_data = re.search(r"TYPE (.+)", response, re.DOTALL).group(1)
        except:
            type_data = re.search(r'TYPE "(.+)"', response, re.DOTALL).group(1)
        return {"type": "TYPE", "data": type_data}

    elif response.startswith("SEARCH"):
        # Extract the search query
        try:
            search_data = re.search(r'SEARCH "(.+)"', response).group(1)
        except:
            search_data = re.search(r"SEARCH (.+)", response).group(1)
        return {"type": "SEARCH", "data": search_data}

    return {"type": "UNKNOWN", "data": response}


def summarize(model, messages, objective):
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "summary_screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        summary_prompt = format_summary_prompt(objective)
        
        if model == "gpt-4-vision-preview":
            with open(screenshot_filename, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            summary_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
            # create a copy of messages and save to pseudo_messages
            messages.append(summary_message)

            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=500,
            )

            content = response.choices[0].message.content
        elif model == "gemini-pro-vision":
            model = genai.GenerativeModel("gemini-pro-vision")
            summary_message = model.generate_content(
                [summary_prompt, Image.open(screenshot_filename)]
            )
            content = summary_message.text
        return content
    
    except Exception as e:
        print(f"Error in summarize: {e}")
        return "Failed to summarize the workflow"

def mouse_click(click_detail):
    return "TRUE"

################################
##START LANG-SOM IMPLEMENTATION#
################################

# Function to calculate center of each box
def calculate_center(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center

# Function to scale coordinates to screen size
def scale_to_screen(x, y, scale_x, scale_y):
    return x * scale_x, y * scale_y

# Scaling factors (from previous calculations)
scale_x = 0.5
scale_y = 0.5

CREATE_DATASET = True

def som_mouse_click(click_detail, messages):
    print("click_detail", click_detail)
    screenshots_dir = "screenshots"
    screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
    
    # Convert screenshot image to base64 for sending
    with open(screenshot_filename, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_bytes = base64.b64decode(img_base64)
        image_object = Image.open(BytesIO(image_bytes))

    textract_response = get_textract_layout_response(image_bytes)

    word_to_coordinates = {}
    for block in textract_response["Blocks"]:
        if block["BlockType"] == "WORD":
            bbox = block["Geometry"]["BoundingBox"]
            x1, y1, x2, y2 = bbox["Left"], bbox["Top"], bbox["Left"] + bbox["Width"], bbox["Top"] + bbox["Height"]
            center = ((x1 + x2) / 2) * image_object.width, ((y1 + y2) / 2) * image_object.height
            word_to_coordinates[block["Text"]] = center

    # Ask model which text it wants to click on
    select_text_prompt = f"Here's what we're trying to do: {click_detail}. Please specify which object you want to click on by picking the text associated with it. For instance to select the safari address bar you would select whatever text is currently within it, i.e. github.com. Text is in order of appereance from left, right, up, down. Return the text and nothing else: " + '\n'.join(word_to_coordinates.keys())
    selected_text_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": select_text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ],
    }

    # Add summary message to existing messages and send to the model
    messages.append(selected_text_message)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500,
        temperature=0
    )

    content = response.choices[0].message.content

    # Get the center of the selected text
    print("\n\n content", content)
    print("\n\n word_to_coordinates", ','.join(word_to_coordinates.keys()))
    selected_text = content.split("\n")[0]
    print("\n\n selected_text", selected_text)
    try:
        x, y = word_to_coordinates[selected_text]
    except:
        selected_text = selected_text.split(" ")[0]
        x, y = word_to_coordinates[selected_text]
    print("\n\n x, y", x, y)

    pyautogui.moveTo(int(x/2), int(y/2), duration=0.2)
    time.sleep(1.5)
    pyautogui.click()
    time.sleep(1.5)
    messages.pop()
    return f"Clicked on text {click_detail['visual_description']}."
    
def segment_image(screenshot_filename, click_detail, segmentation_model):
    image_pil = Image.open(screenshot_filename).convert("RGB")
    draw = ImageDraw.Draw(image_pil)
    font_size = 30 # Adjust as necessary
    font = ImageFont.truetype("Arial.ttf", font_size)

    masks, boxes, phrases, logits = segmentation_model.predict(image_pil, click_detail["visual_description"])
    print(f"Boxes: {boxes}, Masks: {masks}")
    click_coordinates = {}

    # Define a list of color tuples for different masks
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        # Add more colors if needed
    ]

    for i, box in enumerate(boxes):
        center = calculate_center(box)
        screen_x, screen_y = scale_to_screen(center[0], center[1], scale_x, scale_y)

        # Store the screen coordinates for potential clicks, turn them from floats to ints
        click_coordinates[chr(65 + i)] = (int(screen_x), int(screen_y))

        mask = masks[i]
        # Select a color from the list, cycling if necessary
        outline_color = colors[i % len(colors)]

        # Highlighting the mask with a semi-transparent overlay
        overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        binary_mask = mask > 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if binary_mask[y, x]:  # If the mask pixel is 'True'/'1'
                    overlay_draw.point((x, y), fill=(*outline_color, 100))  # Semi-transparent red point

        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), overlay).convert("RGB")

        # Marking the mask with a letter or number
        identifier = chr(65 + i)  # Convert index to ASCII (A, B, C, ...)
        draw.text((center[0], center[1]), identifier, fill=(0, 255, 0), font=font)  # Draw identifier
    return image_pil, click_coordinates

def save_to_dataset(image_pil, click_detail, click_coordinates, content):
    # If dataset directories not created create them
    if not os.path.exists("screenshots/data/images"):
        os.makedirs("screenshots/data/images")
    if not os.path.exists("screenshots/data/masks"):
        os.makedirs("screenshots/data/masks")
    if not os.path.exists("screenshots/data/click_details"):
        os.makedirs("screenshots/data/click_details")
    if not os.path.exists("screenshots/data/predicted_click_coordinates"):
        os.makedirs("screenshots/data/predicted_click_coordinates")
    if not os.path.exists("screenshots/data/predicted_clicks"):
        os.makedirs("screenshots/data/predicted_clicks")
    # Create unique identifier for this click
    id = str(time.time())
    # Save the screenshot to a file in screenshots/dataset/images
    image_pil.save(f"screenshots/data/images/{id}.jpg")
    # Save the masked image file in screenshots/dataset/masks
    image_pil.save(f"screenshots/data/masks/{id}.jpg")
    # Save the click_detail to a file in screenshots/dataset/click_details
    with open(f"screenshots/data/click_details/{id}.json", "w") as f:
        json.dump(click_detail, f)
    # Save the click_coordinates to a file in screenshots/dataset/predicted_click_coordinates
    with open(f"screenshots/data/predicted_click_coordinates/{id}.json", "w") as f:
        json.dump(click_coordinates, f)
    # Save the requested mask click to a file in screenshots/dataset/requested_mask_click
    with open(f"screenshots/data/predicted_clicks/{id}.txt", "w") as f:
        f.write(content)

###############################
##END LANG-SOM IMPLEMENTATION##
###############################

def click_at_percentage(
    x_percentage, y_percentage, duration=0.2, circle_radius=50, circle_duration=0.5
):
    # Get the size of the primary monitor
    screen_width, screen_height = pyautogui.size()

    # Calculate the x and y coordinates in pixels
    x_pixel = int(screen_width * float(x_percentage))
    y_pixel = int(screen_height * float(y_percentage))

    # Move to the position smoothly
    pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

    # Circular movement
    start_time = time.time()
    while time.time() - start_time < circle_duration:
        angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
        x = x_pixel + math.cos(angle) * circle_radius
        y = y_pixel + math.sin(angle) * circle_radius
        pyautogui.moveTo(x, y, duration=0.1)

    # Finally, click
    pyautogui.click(x_pixel, y_pixel)
    return "Successfully clicked"


def add_grid_to_image(original_image_path, new_image_path, grid_interval):
    """
    Add a grid to an image
    """
    # Load the image
    image = Image.open(original_image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Get the image size
    width, height = image.size

    # Reduce the font size a bit
    font_size = int(grid_interval / 10)  # Reduced font size

    # Calculate the background size based on the font size
    bg_width = int(font_size * 4.2)  # Adjust as necessary
    bg_height = int(font_size * 1.2)  # Adjust as necessary

    # Function to draw text with a white rectangle background
    def draw_label_with_background(
        position, text, draw, font_size, bg_width, bg_height
    ):
        # Adjust the position based on the background size
        text_position = (position[0] + bg_width // 2, position[1] + bg_height // 2)
        # Draw the text background
        draw.rectangle(
            [position[0], position[1], position[0] + bg_width, position[1] + bg_height],
            fill="white",
        )
        # Draw the text
        draw.text(text_position, text, fill="black", font_size=font_size, anchor="mm")

    # Draw vertical lines and labels at every `grid_interval` pixels
    for x in range(grid_interval, width, grid_interval):
        line = ((x, 0), (x, height))
        draw.line(line, fill="blue")
        for y in range(grid_interval, height, grid_interval):
            # Calculate the percentage of the width and height
            x_percent = round((x / width) * 100)
            y_percent = round((y / height) * 100)
            draw_label_with_background(
                (x - bg_width // 2, y - bg_height // 2),
                f"{x_percent}%,{y_percent}%",
                draw,
                font_size,
                bg_width,
                bg_height,
            )

    # Draw horizontal lines - labels are already added with vertical lines
    for y in range(grid_interval, height, grid_interval):
        line = ((0, y), (width, y))
        draw.line(line, fill="blue")

    # Save the image with the grid
    image.save(new_image_path)


def keyboard_type(text):
    text = text.replace("\\n", "\n")
    for char in text:
        pyautogui.write(char)
    pyautogui.press("enter")
    return "Type: " + text


def search(text):
    if platform.system() == "Windows":
        pyautogui.press("win")
    elif platform.system() == "Linux":
        pyautogui.press("win")
    else:
        # Press and release Command and Space separately
        pyautogui.keyDown("command")
        pyautogui.press("space")
        pyautogui.keyUp("command")

    time.sleep(1)

    # Now type the text
    for char in text:
        pyautogui.write(char)

    pyautogui.press("enter")
    return "Open program: " + text


def capture_mini_screenshot_with_cursor(
    file_path=os.path.join("screenshots", "screenshot_mini.png"), x=0, y=0
):
    user_platform = platform.system()

    if user_platform == "Linux":
        x = float(x[:-1])  # convert x from "50%" to 50.
        y = float(y[:-1])

        x = (x / 100) * monitor_size[
            "width"
        ]  # convert x from 50 to 0.5 * monitor_width
        y = (y / 100) * monitor_size["height"]

        # Define the coordinates for the rectangle
        x1, y1 = int(x - ACCURATE_PIXEL_COUNT / 2), int(y - ACCURATE_PIXEL_COUNT / 2)
        x2, y2 = int(x + ACCURATE_PIXEL_COUNT / 2), int(y + ACCURATE_PIXEL_COUNT / 2)

        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        screenshot = screenshot.resize(
            (screenshot.width * 2, screenshot.height * 2), Image.LANCZOS
        )  # upscale the image so it's easier to see and percentage marks more visible
        screenshot.save(file_path)

        screenshots_dir = "screenshots"
        grid_screenshot_filename = os.path.join(
            screenshots_dir, "screenshot_mini_with_grid.png"
        )

        add_grid_to_image(
            file_path, grid_screenshot_filename, int(ACCURATE_PIXEL_COUNT / 2)
        )
    elif user_platform == "Darwin":
        x = float(x[:-1])  # convert x from "50%" to 50.
        y = float(y[:-1])

        x = (x / 100) * monitor_size[
            "width"
        ]  # convert x from 50 to 0.5 * monitor_width
        y = (y / 100) * monitor_size["height"]

        x1, y1 = int(x - ACCURATE_PIXEL_COUNT / 2), int(y - ACCURATE_PIXEL_COUNT / 2)

        width = ACCURATE_PIXEL_COUNT
        height = ACCURATE_PIXEL_COUNT
        # Use the screencapture utility to capture the screen with the cursor
        rect = f"-R{x1},{y1},{width},{height}"
        subprocess.run(["screencapture", "-C", rect, file_path])

        screenshots_dir = "screenshots"
        grid_screenshot_filename = os.path.join(
            screenshots_dir, "screenshot_mini_with_grid.png"
        )

        add_grid_to_image(
            file_path, grid_screenshot_filename, int(ACCURATE_PIXEL_COUNT / 2)
        )


def capture_screen_with_cursor(file_path):
    user_platform = platform.system()

    if user_platform == "Windows":
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
    elif user_platform == "Linux":
        # Use xlib to prevent scrot dependency for Linux
        screen = Xlib.display.Display().screen()
        size = screen.width_in_pixels, screen.height_in_pixels
        monitor_size["width"] = size[0]
        monitor_size["height"] = size[1]
        screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
        screenshot.save(file_path)
    elif user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        subprocess.run(["screencapture", "-C", file_path])
    else:
        print(f"The platform you're using ({user_platform}) is not currently supported")


def extract_json_from_string(s):
    # print("extracting json from string", s)
    try:
        # Find the start of the JSON structure
        json_start = s.find("{")
        if json_start == -1:
            return None

        # Extract the JSON part and convert it to a dictionary
        json_str = s[json_start:]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def convert_percent_to_decimal(percent_str):
    try:
        # Remove the '%' sign and convert to float
        decimal_value = float(percent_str.strip("%"))

        # Convert to decimal (e.g., 20% -> 0.20)
        return decimal_value / 100
    except ValueError as e:
        print(f"Error converting percent to decimal: {e}")
        return None
    
def get_textract_layout_response(image_bytes):
    """
    Calls Amazon Textract to get the layout response for the image.

    :param image_bytes: The image bytes to send to Amazon Textract.
    :return: The Amazon Textract layout response.
    """
    textract_client = boto3.client("textract")
    response = textract_client.analyze_document(
        Document={"Bytes": image_bytes}, FeatureTypes=["LAYOUT"]
    )

    return response


def main_entry():
    parser = argparse.ArgumentParser(
        description="Run the self-operating-computer with a specified model."
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to use",
        required=False,
        default="gpt-4-vision-preview",
    )

    # Add a voice flag
    parser.add_argument(
        "--voice",
        help="Use voice input mode",
        action="store_true",
    )

    parser.add_argument(
        "-accurate",
        help="Activate Reflective Mouse Click Mode",
        action="store_true",
        required=False,
    )

    # Add SoM to image
    parser.add_argument(
        "-som",
        help="Activate SOM Mode",
        action="store_true",
        required=False,
    )

    # Allow for direct input of prompt
    parser.add_argument(
        "--prompt",
        help="Directly input the objective prompt",
        type=str,
        required=False,
    )

    try:
        args = parser.parse_args()
        main(
            args.model,
            accurate_mode=args.accurate,
            terminal_prompt=args.prompt,
            voice_mode=args.voice,
            som_mode=args.som,
        )
    except KeyboardInterrupt:
        print(f"\n{ANSI_BRIGHT_MAGENTA}Exiting...")


if __name__ == "__main__":
    main_entry()

