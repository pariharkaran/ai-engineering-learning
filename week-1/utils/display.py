import sys

# Terminal colors using ANSI escape sequences
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

def print_header(title: str):
    """Prints a prominent header block for sections/experiments."""
    border = "=" * 60
    print(f"\n{BLUE}{BOLD}{border}")
    print(f" {title.upper().center(58)}")
    print(f"{border}{RESET}\n")

def print_subheader(title: str):
    """Prints a smaller sub-header separator."""
    print(f"\n{CYAN}{BOLD}--- {title} ---{RESET}")

def print_concept(title: str, explanation: str):
    """
    Prints educational concepts with clean formatting.
    """
    print(f"\n{YELLOW}{BOLD}[EDUCATIONAL CONCEPT] {title}{RESET}")
    formatted = explanation.strip().replace('\n', '\n  ')
    print(f"{BLUE}• {formatted}{RESET}\n")

def print_metric(label: str, value: any, unit: str = ""):
    """Prints a formatted key-value metric."""
    unit_str = f" {unit}" if unit else ""
    print(f"  {BOLD}{label}:{RESET} {GREEN}{value}{unit_str}{RESET}")

def print_info(msg: str):
    """Prints a generic informational message."""
    print(f"{BLUE}[INFO]{RESET} {msg}")

def print_error(msg: str):
    """Prints a clear error message."""
    print(f"\n{RED}{BOLD}[ERROR] {msg}{RESET}\n")

def print_divider():
    """Prints a simple line divider."""
    print(f"{BLUE}-" * 60 + f"{RESET}")
