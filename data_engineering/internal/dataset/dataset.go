package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
)

type Dataset interface {
  LoadLabelsIdx()
}

type dataset struct {
  Labels map[string][]int
  Path   string
}

func NewDataset (path string) Dataset {
  return &dataset{
    Labels: make(map[string][]int),
    Path: path,
  }
}

func (d *dataset) LoadLabelsIdx() {
  
  file, err := os.Open(d.Path)
  if err != nil {
    fmt.Printf("Houve um erro ao abrir o arquivo: %v\n", err)
  }
  defer file.Close()
  
  reader := csv.NewReader(file)  

  data, _ := reader.ReadAll()
  
  for i, row := range data {
    label := row[1]
    d.Labels[label] = append(d.Labels[label], i) 
  }
  
  fmt.Printf("Psoriasis Data: %v\nDermatite Data: %v\n", len(d.Labels["psoriasis"]), len(d.Labels["dermatite"]))
}

// Should save idx to a file
func (d *dataset) SaveIdx() {
  
}
