import fitz
import os

# input_folder = ".//pdfs"
# output_folder = ".//txts"

# os.makedirs(output_folder, exist_ok=True)

def convert_pdf_to_txt(pdf_path, text_path):
    """
    Converts a PDF file to a text file.

    Args:
        pdf_path (str): Path to the PDF file.
        text_path (str): Path to save the converted text file.
    """
    doc = fitz.open(pdf_path)
    with open(text_path, "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text() + '\n')
    doc.close()

def process_pdfs(input_folder, output_folder):
    """
    Processes all PDF files in the input folder and converts them to text files
    in the output folder.

    Args:
        input_folder (str): Path to the folder containing PDF files.
        output_folder (str): Path to the folder to save converted text files.
    """
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_folder)
                text_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(text_subfolder, exist_ok=True)
                text_filename = os.path.splitext(filename)[0] + ".txt"
                text_path = os.path.join(text_subfolder, text_filename)

                if os.path.exists(text_path):
                    print(f"Skipping existing file: {text_path}")
                    continue

                convert_pdf_to_txt(pdf_path, text_path)

# if __name__ == "__main__":
#     process_pdfs(input_folder, output_folder)
#     print(f"PDFs converted to text files in {output_folder}")