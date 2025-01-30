package main

import (
	"flag"
	"os"
	"strings"

	"github.com/checkioname/cnn-skin-model/internal/bgremover"
	"github.com/checkioname/cnn-skin-model/internal/dataset"
	e "github.com/checkioname/cnn-skin-model/internal/utils"
)

func main() {

  path := "/home/king/Downloads/GOMES, FERNANDA SANTOS (20180807152012848) 20200820095642207.jpg"
  
  bgremover.RemoveBlueBg(path)


  // Gerar Dataset
  rootpath := flag.String("p", "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/", "caminho para gerar o dataset")
  gendataset := flag.Bool("d", false, "Gerar ou nao o dataset")
  strat := flag.Bool("s", false, "Gerar indices stratificado do dataset")



  // Parse flags
  flag.Parse()
  

  // Handling flags
  if (*gendataset == true) {
    
    filename := GetSharedFile()
    e.GenerateCsvFromDir(*rootpath, filename)
    println(filename)

  }

  if (*strat == true) {
    filename := GetSharedFile()
    dataset := dataset.NewDataset(filename) 
    dataset.LoadLabelsIdx()
  }


}

func GetSharedFile() string {
    wd, _ := os.Getwd()
    wdname := strings.Split(wd, "/")

    baseDir := strings.Join(wdname[:len(wdname) -1],"/") 
    filename := baseDir + "/dataset.csv"
    return filename
}
