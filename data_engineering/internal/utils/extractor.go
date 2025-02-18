package utils

import (
	"encoding/csv"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

func GenerateCsvFromDir(rootpath, filename string) error {

	var data [][]string

	err := filepath.WalkDir(rootpath, func(path string, d fs.DirEntry, err error) error {
		label := strings.Split(strings.Split(path[len(rootpath):], "/")[0], "_")[0]
		if d.IsDir() == false {
			imagename := d.Name()
			if hasImageExtension(imagename) {
				image_path := path
				data = append(data, []string{image_path, label})

			}
		}
		return nil
	})
	if err != nil {
		fmt.Println("Houve um erro")
	}
	
  file, _ := os.Create(filename)
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

func GetBatches(entries []fs.DirEntry, numbatch int) ([][]fs.DirEntry, error) {

	batchsize := len(entries) / numbatch
	if len(entries)%numbatch != 0 {
		batchsize++
	}

	var batches [][]fs.DirEntry

	for i := 0; i <= len(entries); i += batchsize {
		end := i + batchsize
		if end > len(entries) {
			end = len(entries)
		}
		batches = append(batches, entries[i:end])
	}

	return batches, nil
}
