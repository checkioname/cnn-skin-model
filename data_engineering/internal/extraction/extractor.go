package extraction

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// def generate_csv_from_dir(root_path, output_csv='image_labels.csv'):
//     # Escreve os dados no arquivo CSV
//     with open(output_csv, mode='w', newline='') as file:
//         writer = csv.writer(file)
//         writer.writerow(["img_name", "labels"])
//         print(f"ESCREVENDO OS DADOS {data}")
//         writer.writerows(data)

//     df = read_csv("image_labels.csv")
//     print(df.head())

//     print(f"CSV salvo como {output_csv}")

func GenerateCsvFromDir(rootpath string) error {
	subfolders, _ := getSubfolders(rootpath)
	var data [][]string

	for _, subfolder := range subfolders {
		subfolder_path := filepath.Join(rootpath, subfolder)
		entries, _ := os.ReadDir(subfolder_path)
		for _, entry := range entries {
			imagename := entry.Name()
			if hasImageExtension(imagename) {
				image_path := filepath.Join(subfolder_path, imagename)
				label := subfolder
				data = append(data, []string{image_path, label})
			}
		}
	}

	file, _ := os.Create("datatest.csv")
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{"img_name", "labels"}); err != nil {
		return fmt.Errorf("erro ao escrever cabeçalho no CSV: %w", err)
	}

	if err := writer.WriteAll(data); err != nil {
		return fmt.Errorf("houve um erro ao escrever os dados do dataset: %w", err)
	}

	fmt.Println("Arquivo CSV gerado com sucesso!")
	return nil
}

func getSubfolders(rootpath string) ([]string, error) {
	entries, err := os.ReadDir(rootpath)
	if err != nil {
		return nil, err
	}

	var subfolders []string
	for _, entry := range entries {
		if entry.IsDir() {
			subfolders = append(subfolders, entry.Name())
		}
	}
	return subfolders, nil
}

func hasImageExtension(filename string) bool {
	lowerFilename := strings.ToLower(filename)
	extensions := []string{".png", ".jpg", ".jpeg"}
	for _, ext := range extensions {
		if strings.HasSuffix(lowerFilename, ext) {
			return true
		}
	}
	return false
}
