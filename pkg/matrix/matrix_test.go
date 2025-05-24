package matrix

import (
	"reflect"
	"testing"
)

func TestMultiply(t *testing.T) {
	tests := []struct {
		name    string
		a       [][]float64
		b       [][]float64
		want    [][]float64
		wantErr bool
	}{
		{
			name: "valid 2x2 matrices",
			a:    [][]float64{{1, 2}, {3, 4}},
			b:    [][]float64{{2, 0}, {1, 2}},
			want: [][]float64{{4, 4}, {10, 8}},
			wantErr: false,
		},
		{
			name: "valid 2x3 and 3x2 matrices",
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}},
			b:    [][]float64{{7, 8}, {9, 10}, {11, 12}},
			want: [][]float64{{58, 64}, {139, 154}},
			wantErr: false,
		},
		{
			name: "incompatible dimensions A_cols != B_rows",
			a:    [][]float64{{1, 2}, {3, 4}}, // 2x2
			b:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, // 3x3. A_cols (2) != B_rows (3)
			want: nil,
			wantErr: true,
		},
		{
			name: "A is 1x3, B is 3x1 (vector dot product)",
			a:    [][]float64{{1, 2, 3}},
			b:    [][]float64{{4}, {5}, {6}},
			want: [][]float64{{32}},
			wantErr: false,
		},
		{
			name: "A is 3x1, B is 1x3 (outer product)",
			a:    [][]float64{{1}, {2}, {3}},
			b:    [][]float64{{4, 5, 6}},
			want: [][]float64{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}},
			wantErr: false,
		},
		{
			name: "A is 2x0, B is 0xN (represented as 0x0) -> C is 2x0",
			a:    [][]float64{{}, {}}, // 2x0. rowsA=2, colsA=0
			b:    [][]float64{},    // 0xN (interpreted as 0x0). rowsB=0, colsB=0
			// colsA (0) == rowsB (0). Result is rowsA x colsB = 2x0.
			want: [][]float64{{}, {}}, 
			wantErr: false,
		},
		{
			name: "A is Mx0 (represented as Mx0), B is 0x3 -> C is Mx0",
			a:    [][]float64{{}, {}, {}}, // 3x0. rowsA=3, colsA=0
			b:    [][]float64{nil, nil, nil}, // This represents a 0x3 matrix if b[0] was defined for colsB, but with rowsB=0, it's 0x0.
			// To make B a 0x3 matrix for the Multiply function, it must be passed as [][]float64{}. 
			// Let's re-state this test to be A(MxK) * B(KxN) where K=0.
			// This is covered by the previous test: "A is 2x0, B is 0xN (represented as 0x0) -> C is 2x0"
			// Let's make a new one: A is 2x3, B is 3x0 -> C is 2x0
		},
        {
            name: "A is 2x3, B is 3x0 -> C is 2x0",
            a:    [][]float64{{1,2,3},{4,5,6}}, // 2x3
            b:    [][]float64{{},{},{}}, // 3x0. rowsB=3, colsB=0
            // colsA (3) == rowsB (3). Result is rowsA x colsB = 2x0.
            want: [][]float64{{},{}}, 
            wantErr: false,
        },
		{
			name: "A is 0xK (A empty, 0x0), B is 2x2 -> error",
			a:    [][]float64{}, // 0xK (interpreted as 0x0). rowsA=0, colsA=0
			b:    [][]float64{{1, 2}, {3, 4}}, // 2x2. rowsB=2, colsB=2
			// colsA (0) != rowsB (2). Error.
			want: nil, 
			wantErr: true, 
		},
		{
			name: "A is 2x2, B is 0xK (B empty, 0x0) -> error",
			a:    [][]float64{{1, 2}, {3, 4}}, // 2x2. rowsA=2, colsA=2
			b:    [][]float64{}, // 0xK (interpreted as 0x0). rowsB=0, colsB=0
			// colsA (2) != rowsB (0). Error.
			want: nil, 
			wantErr: true,
		},
        {
            name: "A is 0x0, B is 0x0 -> C is 0x0",
            a:    [][]float64{}, // 0x0
            b:    [][]float64{}, // 0x0
            // colsA (0) == rowsB (0). Result is 0x0.
            want: [][]float64{}, 
            wantErr: false,
        },
	}

	for _, tt := range tests {
        // Skip placeholder test that needs clearer definition for 0xN matrices
        if tt.name == "A is Mx0 (represented as Mx0), B is 0x3 -> C is Mx0" {
            continue
        }
		t.Run(tt.name, func(t *testing.T) {
			got, err := Multiply(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Multiply() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			// If we expect an error, 'got' can be anything (usually nil), so no further checks if err matches wantErr.
			if tt.wantErr {
				return
			}
			// If no error is expected, 'got' must be DeepEqual to 'want'.
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Multiply() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	tests := []struct {
		name    string
		a       [][]float64
		b       [][]float64
		want    [][]float64
		wantErr bool
	}{
		{
			name: "valid 2x2 matrices",
			a:    [][]float64{{1, 2}, {3, 4}},
			b:    [][]float64{{5, 6}, {7, 8}},
			want: [][]float64{{6, 8}, {10, 12}},
			wantErr: false,
		},
		{
			name: "incompatible dimensions (rows differ)",
			a:    [][]float64{{1, 2}, {3, 4}}, // 2x2
			b:    [][]float64{{1, 2}},          // 1x2
			want: nil,
			wantErr: true,
		},
		{
			name: "incompatible dimensions (cols differ)",
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}}, // 2x3
			b:    [][]float64{{1, 2}, {3, 4}},          // 2x2
			want: nil,
			wantErr: true,
		},
		{
			name: "both 0x0 matrices",
			a:    [][]float64{},
			b:    [][]float64{},
			want: [][]float64{},
			wantErr: false,
		},
		{
			name: "both 2x0 matrices",
			a:    [][]float64{{}, {}},
			b:    [][]float64{{}, {}},
			want: [][]float64{{}, {}},
			wantErr: false,
		},
		{
			name: "A is 2x0, B is 0x0 -> error",
			a:    [][]float64{{}, {}}, // 2x0
			b:    [][]float64{},       // 0x0
			want: nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Add(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Add() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSubtract(t *testing.T) {
	tests := []struct {
		name    string
		a       [][]float64
		b       [][]float64
		want    [][]float64
		wantErr bool
	}{
		{
			name: "valid 2x2 matrices",
			a:    [][]float64{{5, 8}, {10, 12}},
			b:    [][]float64{{1, 2}, {3, 4}},
			want: [][]float64{{4, 6}, {7, 8}},
			wantErr: false,
		},
		{
			name: "incompatible dimensions (rows differ)",
			a:    [][]float64{{1, 2}, {3, 4}}, // 2x2
			b:    [][]float64{{1, 2}},          // 1x2
			want: nil,
			wantErr: true,
		},
		{
			name: "incompatible dimensions (cols differ)",
			a:    [][]float64{{1, 2, 3}, {4, 5, 6}}, // 2x3
			b:    [][]float64{{1, 2}, {3, 4}},          // 2x2
			want: nil,
			wantErr: true,
		},
		{
			name: "both 0x0 matrices",
			a:    [][]float64{},
			b:    [][]float64{},
			want: [][]float64{},
			wantErr: false,
		},
		{
			name: "both 2x0 matrices",
			a:    [][]float64{{}, {}},
			b:    [][]float64{{}, {}},
			want: [][]float64{{}, {}},
			wantErr: false,
		},
		{
			name: "A is 2x0, B is 0x0 -> error",
			a:    [][]float64{{}, {}}, // 2x0
			b:    [][]float64{},       // 0x0
			want: nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Subtract(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Subtract() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Subtract() got = %v, want %v", got, tt.want)
			}
		})
	}
}
