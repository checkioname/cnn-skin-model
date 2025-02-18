package main

import (
	"flag"
	"fmt"
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
	"github.com/checkioname/cnn-skin-model/internal/utils"
)

func main() {

	// Definir subcomandos
	genDatasetCmd := flag.NewFlagSet("gendataset", flag.ExitOnError)
	removeBgCmd := flag.NewFlagSet("removebg", flag.ExitOnError)
	stratCmd := flag.NewFlagSet("stratify", flag.ExitOnError)
	augmentCmd := flag.NewFlagSet("augment", flag.ExitOnError)

	// Flags do subcomando `gendataset`
	rootPath := genDatasetCmd.String("p", "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/", "Caminho do dataset")

	// Flags do subcomando `removebg`
	rmDir := removeBgCmd.String("indir", "", "Diretório com imagens para remover background")
	rmOutDir := removeBgCmd.String("outdir", "", "Diretório para salvar imagens sem background")

	// Flags do subcomando `augment`
	augDir := augmentCmd.String("indir", "", "Diretório de entrada com imagens para aumento de base")
	augOutDir := augmentCmd.String("outdir", "", "Diretório para salvar imagens aumentadas")

	flag.Parse()

	switch os.Args[1] {
	case "gendataset":
		filename := GetSharedFile()
		genDatasetCmd.Parse(os.Args[2:])
		utils.GenerateCsvFromDir(*rootPath, filename)

	case "removebg":
		removeBgCmd.Parse(os.Args[2:])
		if *rmDir == "" || *rmOutDir == "" {
			fmt.Println("Erro: --indir e --outdir são obrigatórios para removebg")
			os.Exit(1)
		}

		blueRemover := bgremover.NewBgRemover("blue")
		bgremover.RemoveBlueBgDir(*rmDir, *rmOutDir, blueRemover)
	case "stratify":
		stratCmd.Parse(os.Args[2:])

		filename := GetSharedFile()
		dataset := dataset.NewDataset(filename)
		dataset.LoadLabelsIdx()

	case "augment":
		augmentCmd.Parse(os.Args[2:])
		if *augDir == "" || *augOutDir == "" {
			fmt.Println("Erro: --indir e --outdir são obrigatórios para augment")
			os.Exit(1)
		}
		resize := resize.NewResize()
		bright := brightness.NewBrightness()
		denoise := denoise.NewDenoise()
		flip := flip.NewFlip()
		rotate := rotate.NewRotate()

		augmentation.BatchAugmentation(*augDir, *augOutDir, resize, bright, denoise, flip, rotate)
	default:
		fmt.Println("Subcomando inválido. Use: gendataset | removebg | stratify | augment")
		os.Exit(1)
	}
}

func GetSharedFile() string {
	wd, _ := os.Getwd()
	wdname := strings.Split(wd, "/")

	baseDir := strings.Join(wdname[:len(wdname)-1], "/")
	filename := baseDir + "/dataset.csv"
	return filename
}
