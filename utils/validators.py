# ============================================================
# utils/validators.py
# ============================================================
# Input validation functions used by the UI layer.
# Does NOT call any LLM or import from Streamlit.
# ============================================================

from config import MAX_UPLOAD_MB, SUPPORTED_FILE_TYPES


def validate_uploaded_files(uploaded_files: list) -> tuple[bool, list[str]]:
    """
    Validates a list of uploaded files before processing.

    Checks:
    - At least one file uploaded
    - All files are supported type (PDF)
    - No individual file exceeds size limit
    - Total combined size does not exceed limit

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of (is_valid: bool, error_messages: list[str])
    """
    errors = []

    if not uploaded_files:
        return False, ["Please upload at least one PDF file."]

    total_size_mb = 0.0

    for f in uploaded_files:
        # Check file extension
        ext = f.name.split(".")[-1].lower()
        if ext not in SUPPORTED_FILE_TYPES:
            errors.append(
                f"{f.name} is not a supported file type. "
                f"Only PDF files are accepted."
            )

        # Check individual file size
        file_size_mb = f.size / (1024 * 1024)
        total_size_mb += file_size_mb

        if file_size_mb > MAX_UPLOAD_MB:
            errors.append(
                f"{f.name} is {file_size_mb:.1f}MB — "
                f"exceeds the {MAX_UPLOAD_MB}MB limit per file."
            )

    # Check total combined size
    if total_size_mb > MAX_UPLOAD_MB:
        errors.append(
            f"Total upload size is {total_size_mb:.1f}MB — "
            f"exceeds the {MAX_UPLOAD_MB}MB combined limit."
        )

    return len(errors) == 0, errors


def validate_summary_inputs(
    max_words: int,
    output_format: str,
) -> tuple[bool, list[str]]:
    """
    Validates summarization inputs from the UI.

    Args:
        max_words: Word limit from slider.
        output_format: Format key from dropdown.

    Returns:
        Tuple of (is_valid: bool, error_messages: list[str])
    """
    from config import SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS, SUMMARY_FORMATS
    errors = []

    if not (SUMMARY_MIN_WORDS <= max_words <= SUMMARY_MAX_WORDS):
        errors.append(
            f"Word limit must be between {SUMMARY_MIN_WORDS} "
            f"and {SUMMARY_MAX_WORDS}."
        )

    if output_format not in SUMMARY_FORMATS:
        errors.append(
            f"Invalid output format: {output_format}."
        )

    return len(errors) == 0, errors