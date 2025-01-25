package main

import (
	"flag"
	"os"
	"strings"

	e "github.com/checkioname/cnn-skin-model/internal/utils"
)

func main() {

  // Gerar Dataset
  rootpath := flag.String("p", "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/", "caminho para gerar o dataset")
  gendataset := flag.Bool("d", false, "Gerar ou nao o dataset")


  // Parse flags
  flag.Parse()
  



  // Handling flags
  if (*gendataset == true) {
    
    wd, _ := os.Getwd()
    baseDir := strings.Join(strings.Split(wd, "/")[:len(strings.Split(wd, "/")) -1],"/") 

    filename := baseDir + "/model_engineering/dataset.csv"
    e.GenerateCsvFromDir(*rootpath, filename)
    println(filename)

  }

}
