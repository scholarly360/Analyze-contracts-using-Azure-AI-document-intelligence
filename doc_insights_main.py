


from doc_insights_utils import * 
from dotenv import load_dotenv

### convert the pdf to png
pdf_to_png('./sample-contract.pdf', './tmp_folder')

### get doc ocr text
client = get_client()
list_whole_content_from_doc_intelligence = analyze_files_in_directory(client, './tmp_folder')

### classify the text
system_instruction = """Given the text below, find out if it belongs to a specific category.\ncategory: early termination\nAnswer only in 'yes' or 'no'"""
list_bboxes = create_bounding_boxes_if_classification(list_whole_content_from_doc_intelligence,system_instruction)

### save
mark_output(list_bboxes, './tmp_folder', './final_output')
##delete_png_files('./tmp_folder')