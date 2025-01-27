package stratkfold

import (
	"math/rand"
	"errors"
)



type Stratkfold struct {
  NSplits    int
  Shuffle    bool
  RandomSeed int64 
}


func NewStratifiedKFold(nsplits int, shuffle bool, seed int64) (*Stratkfold, error){
  if nsplits < 2 {
    return &Stratkfold{}, errors.New("nsplits deve ser pelo menos 2")
  } 

  return &Stratkfold{
    NSplits: nsplits,
    Shuffle: shuffle,
    RandomSeed: seed,
  }, nil
}


func (sk *Stratkfold) Split(data map[string][]int) {
  if sk.Shuffle {
    for _, indices := range data {
      rand.Shuffle(len(indices), func(i,j int ) {
        indices[i], indices[j] = indices[j], indices[i]
      })
    }
  }
}
