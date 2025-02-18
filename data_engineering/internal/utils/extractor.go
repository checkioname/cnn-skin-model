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
    if !d.IsDir() {
      relativePath := strings.TrimPrefix(path, rootpath)
			labelPath := strings.Split(relativePath, "/")
			label := "UNKNOWN"

			if len(labelPath) > 1 && labelPath[1] != "" {
				dirName := strings.ToLower(labelPath[1])
				switch {
				case strings.HasPrefix(dirName, "psoriasis"):
					label = "psoriasis"
				case strings.HasPrefix(dirName, "dermatite"):
					label = "dermatite"
				}
			}

    if hasImageExtension(d.Name()) {
				imagePath := path
				data = append(data, []string{imagePath, label})
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

func GetBatches(entries []string, numbatch int) ([][]string, error) {

	batchsize := len(entries) / numbatch
	if len(entries)%numbatch != 0 {
		batchsize++
	}

	var batches [][] string

	for i := 0; i <= len(entries); i += batchsize {
		end := i + batchsize
		if end > len(entries) {
			end = len(entries)
		}
		batches = append(batches, entries[i:end])
	}

	return batches, nil
}


func ReadNestedDir(path string) []string {
  var fullpath []string

	err := filepath.WalkDir(path, func(path string, d fs.DirEntry, err error) error {
    fullpath = append(fullpath, path)
		return nil
	})
  
  if err != nil {
    return nil
  }

  return fullpath
}


func CreateNestedDir(path string) {
}
