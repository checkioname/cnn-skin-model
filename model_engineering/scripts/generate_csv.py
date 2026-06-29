import os
import csv
import argparse


def find_images(root_dir):
    extensions = {'.jpg', '.jpeg', '.png'}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in extensions:
                yield os.path.relpath(os.path.join(dirpath, f), root_dir)


def main():
    parser = argparse.ArgumentParser(description='Gera CSV a partir de diretório de imagens')
    parser.add_argument('root_dir', help='Diretório raiz com subpastas dermatite/ e psoriasis/')
    parser.add_argument('-o', '--output', default='dataset.csv', help='Arquivo CSV de saída')
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    classes = {'dermatite', 'psoriasis'}
    rows = []

    for rel_path in find_images(root):
        parts = rel_path.replace('\\', '/').split('/')
        if parts[0] in classes:
            rows.append({'img_name': rel_path.replace('\\', '/'), 'labels': parts[0]})

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
