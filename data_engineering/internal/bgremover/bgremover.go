package bgremover

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"regexp"
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
	var entries []fs.DirEntry
  var fullpath []string

	err := filepath.WalkDir(inpDir, func(path string, d fs.DirEntry, err error) error {
		entries = append(entries, d)
    fullpath = append(fullpath, path)
		return nil
	})

	if err != nil {
		fmt.Println("Houve um erro na listagem do diretorio")
		return
	}

	batches, _ := utils.GetBatches(fullpath, 2)
	fmt.Println(batches)
  
	var wg sync.WaitGroup

	for _, batch := range batches {
		wg.Add(1)
		go func([]string) {
			defer wg.Done()

			for _, entry := range batch {
        if match, _ := regexp.Match(".jpg", []byte(entry)); match == false {
					fmt.Println(entry)
					continue
				}
        filename := strings.Split(entry, "/")[len(entry)-1]
        outpath := filepath.Join(outDir, filename)
				outpath = strings.Replace(outpath, ".jpg", "_no_bg.jpg", 1)

				imgfullpath := filepath.Join(inpDir, filename)
				fmt.Println(imgfullpath)
        continue 
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
