import os

def print_structure(root_dir, max_depth=2):
    for root, dirs, files in os.walk(root_dir):
        # Bereken de huidige diepte
        level = root.replace(root_dir, '').count(os.sep)
        if level > max_depth:
            # Verwijder verdere directories
            dirs[:] = []
            continue
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == "__main__":
    project_dir = '.'  # Huidige map
    print_structure(project_dir, max_depth=2)
