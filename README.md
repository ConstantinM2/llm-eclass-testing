# Unlocking Semantic Interoperability in Industry with Large Language Models

Presents a novel conceptual framework for enhancing semantic interoperability in Asset Administration Shells using Large Language Models (LLMs) and ECLASS.

## Description

see Abstract

## Getting Started

### Dependencies

The GUI uses Python with several third party libraries. Install them in a virtual
environment with pip:

```bash
python3 -m pip install PyPDF2 pdfplumber keybert sentence-transformers torch
```

The application relies on a graphical desktop environment for Tkinter. Make sure
you run it on a system that can display windows (e.g. Windows, macOS or Linux
with X11/Wayland).

### Installing

Clone this repository or download its ZIP archive and extract it. No additional
configuration is required; the example PDFs and the ECLASS JSON file are already
included.

### Executing program

1. Launch the GUI:
   ```bash
   python pdf-processing-tool.py
   ```
2. Click **"Open PDF"** and select one of the provided data sheets
   (`D7960_N-en.pdf` or `HY30-2800-UK.pdf`).
3. Choose exactly one suggested ECLASS category and press **"Next"**.
4. The tool extracts technical terms with KeyBERT and filters them by semantic
   similarity to the selected category features. Uncheck unwanted terms and save
   the remaining ones to a text file.
5. Alternatively run the command-line tool for headless environments:
   ```bash
   python pdf_processing_cli.py D7960_N-en.pdf
   ```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```



## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

