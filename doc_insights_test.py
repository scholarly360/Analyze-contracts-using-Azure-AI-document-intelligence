
from doc_insights_utils import * 
from dotenv import load_dotenv

#pdf_to_png('./sample-contract.pdf', './tmp_folder')

# list_png_files = get_png_files('./tmp_folder')
# print(list_png_files)


# system_instruction = """Given the text below, find out if it belongs to a specific category.\ncategory: insurance\nAnswer only in 'yes' or 'no'"""
# para_text = """The Parties agree to the following terms and conditions:"""
# resultant = my_custom_classifier(system_instruction, para_text)
# print(resultant)

# system_instruction = """Given the text below, find out if it belongs to a specific category.\ncategory: insurance\nAnswer only in 'yes' or 'no'"""
# para_text = """Insurance is discussed in the following terms and conditions:"""
# resultant = my_custom_classifier(system_instruction, para_text)
# print(resultant)

client = get_client()
list_whole_content_from_doc_intelligence = analyze_files_in_directory(client, './tmp_folder')
#print(list_whole_content_from_doc_intelligence)

# import json
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(list_whole_content_from_doc_intelligence, f, ensure_ascii=False, indent=4)


system_instruction = """Given the text below, find out if it belongs to a specific category.\ncategory: early termination\nAnswer only in 'yes' or 'no'"""
list_bboxes = create_bounding_boxes_if_classification(list_whole_content_from_doc_intelligence,system_instruction)
#print(list_bboxes)


mark_output(list_bboxes, './tmp_folder', './final_output')