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
	// (skin seg)
	// bgremover.RemoveBlueBg(path)
	// bgremover.SegmentateSkin(path)

	// Gerar Dataset
	rootpath := flag.String("p", "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/", "caminho para gerar o dataset")
	gendataset := flag.Bool("d", false, "Gerar ou nao o dataset")
	strat := flag.Bool("s", false, "Gerar indices stratificado do dataset")

	rmbg := flag.Bool("r", false, "Remover background do diretorio")

	// Parse flags
	flag.Parse()

	// Handling flags
	if *gendataset == true {

		filename := GetSharedFile()
		e.GenerateCsvFromDir(*rootpath, filename)
		println(filename)

	}
	if *rmbg {
		dir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/psoriasis/"
		outdir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/nobluebg_test/"
    
    //choose methods that implement remove bg interface (blue remover, sking segmentation remover)
    blueRemover := bgremover.NewBgRemover("blue")
    bgremover.RemoveBlueBgDir(dir, outdir, blueRemover)
	}

	if *strat == true {
		filename := GetSharedFile()
		dataset := dataset.NewDataset(filename)
		dataset.LoadLabelsIdx()
	}

}

func GetSharedFile() string {
	wd, _ := os.Getwd()
	wdname := strings.Split(wd, "/")

	baseDir := strings.Join(wdname[:len(wdname)-1], "/")
	filename := baseDir + "/dataset.csv"
	return filename
}
