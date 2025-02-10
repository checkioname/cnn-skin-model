package bgremover

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/checkioname/cnn-skin-model/internal/utils"
	"gocv.io/x/gocv"
)

type BgRemover interface {
	RemoveBG(imgPath string) (gocv.Mat, error)
}

func NewBgRemover(method string) BgRemover {
	if method == "blue" {
		return &BlueBgRemover{}
	}
	if method == "skin" {
		return &SkinBgRemover{}
	}
	return nil
}

// batch removing
func RemoveBlueBgDir(inpDir, outDir string, bgRemover BgRemover) {
	entries, err := os.ReadDir(inpDir)
	if err != nil {
		fmt.Println("Houve um erro na listagem do diretorio")
		fmt.Println(err)
		return
	}

	batches, _ := utils.GetBatches(entries, 2)

	var wg sync.WaitGroup

	for _, batch := range batches {
		wg.Add(1)
		go func([]fs.DirEntry) {
			defer wg.Done()

			for _, entry := range batch {
				outpath := filepath.Join(outDir, entry.Name())
				outpath = strings.Replace(outpath, ".jpg", "_no_bg.jpg", 1)

				imgfullpath := filepath.Join(inpDir, entry.Name())
				result, _ := bgRemover.RemoveBG(imgfullpath)
				defer result.Close()
				if result.Empty() {
					fmt.Println("O resultado da transformacao eh vazio")
					break
				}
				gocv.IMWrite(outpath, result)
			}
		}(batch)
	}

	wg.Wait()
}
