import os
import csv
import argparse


CLASS_MAP = {
    'dermatite': 'dermatite',
    'dermatite_atopica': 'dermatite',
    'psoriasis': 'psoriasis',
    'psoriase': 'psoriasis',
    'psoriase_vulgar': 'psoriasis',
}


def find_images(root_dir):
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in extensions:
                yield os.path.relpath(os.path.join(dirpath, f), root_dir)


def main():
    parser = argparse.ArgumentParser(description='Gera CSV a partir de diretorio de imagens')
    parser.add_argument('root_dir', help='Diretorio raiz com subpastas de classes')
    parser.add_argument('-o', '--output', default='dataset.csv', help='Arquivo CSV de saida')
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    rows = []

    for rel_path in find_images(root):
        parts = rel_path.replace('\\', '/').split('/')
        class_name = parts[0].lower()
        if class_name in CLASS_MAP:
            rows.append({'img_name': rel_path.replace('\\', '/'), 'labels': CLASS_MAP[class_name]})

    if not rows:
        print(f"Nenhuma imagem encontrada em {root}")
        return

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_name', 'labels'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV gerado: {args.output} ({len(rows)} amostras)")


if __name__ == '__main__':
    main()
