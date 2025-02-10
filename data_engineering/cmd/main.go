package main

import (
	"flag"
	"os"
	"strings"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"github.com/checkioname/cnn-skin-model/internal/augmentation/brightness"
	"github.com/checkioname/cnn-skin-model/internal/augmentation/denoise"
	"github.com/checkioname/cnn-skin-model/internal/augmentation/flip"
	"github.com/checkioname/cnn-skin-model/internal/augmentation/resize"
	"github.com/checkioname/cnn-skin-model/internal/augmentation/rotate"
	"github.com/checkioname/cnn-skin-model/internal/bgremover"
	"github.com/checkioname/cnn-skin-model/internal/dataset"
	e "github.com/checkioname/cnn-skin-model/internal/utils"
)

func main() {
	// flags
	// Gerar Dataset
	rootpath := flag.String("p", "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/", "caminho para gerar o dataset")
	gendataset := flag.Bool("d", false, "Gerar ou nao o dataset")
	strat := flag.Bool("s", false, "Gerar indices stratificado do dataset")

	rmbg := flag.Bool("r", false, "Remover background do diretorio")
	augbg := flag.Bool("a", false, "Aumento das imagens do diretorio")
	// Parse flags
	flag.Parse()

	// Handling flags
	if *gendataset {

		filename := GetSharedFile()
		e.GenerateCsvFromDir(*rootpath, filename)
		println(filename)

	}

	if *rmbg {
		dir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/dermatite/"
		outdir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/noblue_bg_dermatite/"

		//choose methods that implement remove bg interface (blue remover, sking segmentation remover)
		blueRemover := bgremover.NewBgRemover("blue")
		bgremover.RemoveBlueBgDir(dir, outdir, blueRemover)
	}

	if *strat {
		filename := GetSharedFile()
		dataset := dataset.NewDataset(filename)
		dataset.LoadLabelsIdx()
	}
	*augbg = true
	if *augbg {
		dir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/db_no_bluebg/noblue_bg_dermatite/"
		//"/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/dermatite/"
		outdir := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/db_augmented/dermatite/"
		resize := resize.NewResize()
		bright := brightness.NewBrightness()
		denoise := denoise.NewDenoise()
		flip := flip.NewFlip()
		rotate := rotate.NewRotate()

		augmentation.BatchAugmentation(dir, outdir, resize, bright, denoise, flip, rotate)
	}

}

func GetSharedFile() string {
	wd, _ := os.Getwd()
	wdname := strings.Split(wd, "/")

	baseDir := strings.Join(wdname[:len(wdname)-1], "/")
	filename := baseDir + "/dataset.csv"
	return filename
}
