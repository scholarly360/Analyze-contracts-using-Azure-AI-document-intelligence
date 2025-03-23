import os
import glob
from pdf2image import convert_from_path
import base64
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI  
from PIL import Image, ImageDraw

"""  pdf_to_png function converts a PDF file to a list of PNG images."""
def pdf_to_png(pdf_path, out_tmp_path):
    # Ensure the output directory exists
    os.makedirs(out_tmp_path, exist_ok=True)
    # Convert PDF to list of images
    images = convert_from_path(pdf_path,dpi=200)
    # Save images as PNG files in the specified output path
    for i, image in enumerate(images):
        output_file = os.path.join(out_tmp_path, f'page_{i+1}.png')
        image.save(output_file, 'PNG')
        print(f"Saved: {output_file}")

"""  get_png_files function returns a list of PNG files in the specified directory."""
def get_png_files(directory):
    try:
        png_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith('.png')
        ]
        return png_files
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
    
"""  Create a pattern to find all .png files and dlete them from the directory."""
def delete_png_files(directory):
    
    pattern = os.path.join(directory, '*.png')
    png_files = glob.glob(pattern)

    if not png_files:
        print("No PNG files found in the directory.")
        return

    for file in png_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def get_client():
    load_dotenv()
    endpoint = os.getenv('azure_doc_endpoint')
    key = os.getenv('azure_doc_key')
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return(client)

""" single file analysis """
def analyze_local_file(client, file_path):
    try:
        with open(file_path, "rb") as f:
            base64_encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
        
        analyze_request = {
            "base64Source": base64_encoded_pdf
        }

        poller = client.begin_analyze_document("prebuilt-layout", analyze_request)
        result = poller.result()

        return result.as_dict()

    except Exception as e:
        print(f"An error occurred processing {file_path}: {e}")
        return None

""" single file analysis to dir level  """
def analyze_files_in_directory(client, directory_path):
    analysis_results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f"Analyzing {file_path}...")
            result_dict = analyze_local_file(client, file_path)
            if result_dict is not None:
                analysis_results.append({
                    "file_name": file_path,
                    "analysis_result": result_dict
                })
    return analysis_results


import shutil

def copy_files(source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all files in the source folder
    files = os.listdir(source_folder)

    for file_name in files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        # Copy only if it's a file
        if os.path.isfile(source_path):
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file_name}")
        else:
            print(f"Skipped (not a file): {file_name}")


def my_custom_classifier(custom_system_instruction,custom_para_text):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
    deployment = os.getenv("AZURE_MODEL", "gpt-4o-mini")  
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")  

    # Initialize Azure OpenAI Service client with key-based authentication    
    az_client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=subscription_key,  
        api_version="2024-05-01-preview",
    )
        
        
    # IMAGE_PATH = "YOUR_IMAGE_PATH"
    # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

    #Prepare the chat prompt 
    chat_prompt = [
        {
            "role": "system","content": [{"type": "text","text": custom_system_instruction}],
        },
        {
            "role": "user","content": [{"type": "text","text": custom_para_text}],
        }
    ] 
        
    # Include speech result if speech is enabled  
    messages = chat_prompt  
        
    # Generate the completion  
    completion = az_client.chat.completions.create(model=deployment,
        messages=messages,
        max_tokens=20,  
        temperature=0.1,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False,
    )
    return(completion.to_dict()["choices"][0]['message']['content'])
    

# Function to create bounding boxes for the content
def create_bounding_boxes_if_classification(list_whole_content_from_doc_intelligence, custom_system_instruction):
    bboxes = []
    for pg_no,whole_content_from_doc_intelligence in enumerate(list_whole_content_from_doc_intelligence):
        paragraphs = whole_content_from_doc_intelligence['analysis_result']['paragraphs']
        file_name = whole_content_from_doc_intelligence['file_name']
        for paragraph in paragraphs:
            content = paragraph['content']
            bounding_regions = paragraph['boundingRegions']
            for region in bounding_regions:
                resultant_text = my_custom_classifier(custom_system_instruction,content)
                if('yes' in resultant_text.lower()):
                    bbox = {'content': content,'pageNumber': pg_no+1,'file_name':file_name, 'polygon': region['polygon']}
                    bboxes.append(bbox)
    return bboxes


def mark_output(list_bboxes, input_folder, output_folder):
    delete_png_files("./final_output")

    png_files = get_png_files(input_folder)
    if not png_files:
        print("No PNG files found in the input folder.")
        return
    
    for item in list_bboxes:
        file_name = item['file_name']
        page_number = item['pageNumber']
        polygon = item['polygon']
        with Image.open(file_name) as img:
            draw = ImageDraw.Draw(img)
            bbox_processed = (polygon[0],polygon[1],polygon[4],polygon[5])
            draw.rectangle(bbox_processed, outline=(255, 69, 0), width=5)
            img.save(file_name)

    copy_files(input_folder, output_folder)
