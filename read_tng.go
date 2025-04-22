package main

import (	
	"C"
	"unsafe"
)

const (
	MinPairs = 3 // Minimum number of pairs needed match up two branches
	ManualLen = 25 // When quick-sort switches to Shell's sort
)

var (
	pairCounts map[uint64]uint8
	max1, max2 int64
)

//export ResetPairCounts
func ResetPairCounts() {
	pairCounts = map[uint64]uint8{ }
	max1, max2 = 0, 0
}


//export IDToIndex
func IDToIndex(IDs, idx *C.longlong, nID, nIndex C.longlong) {
	IDsSlice := unsafe.Slice((*int64)(unsafe.Pointer(IDs)), int(nID))
	idxSlice := unsafe.Slice((*int64)(unsafe.Pointer(idx)), int(nIndex))
	idToIndex(IDsSlice, idxSlice)
}

func idToIndex(IDs, idx []int64) {
	for i, id := range IDs {
		if id != -1 {
			idx[id] = int64(i)
		}
	}
}

//export AddPairs
func AddPairs(
	idx1, idx2 *C.longlong, n C.longlong,
) {
	idx1Slice := unsafe.Slice((*int64)(unsafe.Pointer(idx1)), int(n))
	idx2Slice := unsafe.Slice((*int64)(unsafe.Pointer(idx2)), int(n))

	addPairs(idx1Slice, idx2Slice)
}

func addPairs(idx1, idx2 []int64) {
	for i := range idx1 {
		if idx1[i] > max1 { max1 = idx1[i]+1 }
		if idx2[i] > max2 { max2 = idx2[i]+1 }

		if idx1[i] == -1 || idx2[i] == -1 { continue }
		tag := (uint64(idx1[i]) << 32) | uint64(idx2[i])
		if _, ok := pairCounts[tag]; ok {
			pairCounts[tag]++
		} else {
			pairCounts[tag] = 1
		}
	}
}

//export MatchPairs
func MatchPairs(from1To2 *C.longlong, n C.longlong) {
	from1To2Slice := unsafe.Slice((*int64)(unsafe.Pointer(from1To2)), int(n))
	matchPairs(from1To2Slice)
}

func matchPairs(from1To2 []int64) {
	for i := range from1To2 { from1To2[i] = -1 }
	// I can't believe I missed those +1's for this many years.
	used1, used2 := make([]bool, max1+1), make([]bool, max2+1)

	n := len(pairCounts)
	idx1, idx2, counts := make([]int, n), make([]int, n), make([]int, n)
	i := 0
	for tag, count := range pairCounts {
		idx1[i], idx2[i] = int(tag >> 32), int(tag & 0xffffffff)
		counts[i] = int(count)
		i++
	}

	order := QuickSortIndex(counts)
	max := order[0] 
	for i := range order {
		if order[i] > max { max = order[i] }
	}

	for i = len(order) - 1; i >= 0; i-- {
		j := order[i]
		if counts[j] < MinPairs { break }

		i1, i2 := idx1[j], idx2[j]
		if used1[i1] || used2[i2] { continue }
		used1[i1], used2[i2] = true, true
		from1To2[i1] = int64(i2)
	}
}

// QuickSortIndex returns the indices of the array elements after they've been
// sorted in ascending order.
func QuickSortIndex(xs []int) []int {
	xCopy := make([]int, len(xs))
	for i := range xCopy { xCopy[i] = xs[i] }
	
	idx := make([]int, len(xs))
	for i := range idx { idx[i] = i }
	quickIndex(xCopy, idx)

	return idx
}

func sort3Index(x, y, z, ix, iy, iz int) (
	max, mid, min, maxi, midi, mini int,
) {
	if x > y {
		if x > z {
			if y > z {
				return x, y, z, ix, iy, iz
			} else {
				return x, z, y, ix, iz, iy
			}
		} else {
			return z, x, y, iz, ix, iy
		}
	} else {
		if y > z {
			if x > z {
				return y, x, z, iy, ix, iz
			} else {
				return y, z, x, iy, iz, ix
			}
		} else {
			return z, y, x, iz, iy, ix
		}
	}
}


func quickIndex(xs, idx []int) {
	if len(idx) < ManualLen {
		shellSortIndex(xs, idx)
	} else {
		pivIdx := partitionIndex(xs, idx)
		quickIndex(xs, idx[0:pivIdx])
		quickIndex(xs, idx[pivIdx:len(idx)])
	}	
}

func partitionIndex(xs, idx []int) int {
	n, n2 := len(idx), len(idx)/2
	// Take three values. The median will be the pivot, the other two will
	// be sentinel values so that we can avoid bounds checks.
	_, _, _, maxi, midi, mini := sort3Index(
		xs[idx[0]], xs[idx[n2]], xs[idx[n-1]],
		idx[0], idx[n2], idx[n-1],
	)

	idx[0], idx[n2], idx[n-1] = mini, midi, maxi
	idx[1], idx[n2] = idx[n2], idx[1]
	lo, hi := 1, n-1
	for {
		lo++
		for xs[idx[lo]] < xs[midi] {
			lo++
		}
		hi--
		for xs[idx[hi]] > xs[midi] {
			hi--
		}
		if hi < lo {
			break
		}
		idx[lo], idx[hi] = idx[hi], idx[lo]
	}

	// Swap the pivot into the middle
	idx[1], idx[hi] = idx[hi], idx[1]

	return hi
}

func ShellSortIndex(xs []int) []int {
	idx := make([]int, len(xs))
	for i := range idx { idx[i] = i }
	shellSortIndex(xs, idx)
	return idx
}

// shellSortIndex does an in-place Shell sort of idx such that xs[idx[i]] is
// sorted in ascending order.
func shellSortIndex(xs []int, idx []int) {
	n := len(idx)
	if n == 1 { return }

	inc := 1
	for inc <= n { inc = inc*3 + 1 }

	for inc > 1 {
		inc /= 3
		for i := inc; i < n; i++ {
			v := xs[idx[i]]
			vi := idx[i]
			
			j := i
			for xs[idx[j-inc]] > v {
				idx[j] = idx[j - inc]
				j -= inc
				if j < inc { break }
			}
			idx[j] = vi
		}
	}
	return
}


func main() {
	
}
