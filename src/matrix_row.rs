//! row-oriented matrix to build a problem constraint by constraint
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::matrix_col::ColMatrix;
use crate::{Problem, Row};

/// Represents a variable
pub type Col = usize;

/// A complete optimization problem stored by row
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowMatrix {
    /// column-wise sparse constraints  matrix
    /// Each element in the outer vector represents a column (a variable)
    columns: Vec<(Vec<c_int>, Vec<f64>)>,
}

/// Functions to use when first declaring variables, then constraints.
impl Problem<RowMatrix> {
    /// add a variable to the problem.
    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) -> Col {
        self.add_column_with_integrality(col_factor, bounds, false)
    }

    /// Same as add_column, but forces the solution to contain an integer value for this variable.
    pub fn add_integer_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) -> Col {
        self.add_column_with_integrality(col_factor, bounds, true)
    }

    /// Same as add_column, but lets you define whether the new variable should be integral or continuous.
    #[inline]
    pub fn add_column_with_integrality<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
        is_integer: bool,
    ) -> Col {
        let col = self.num_cols();
        self.add_column_inner(col_factor, bounds, is_integer);
        self.matrix.columns.push((vec![], vec![]));
        col
    }

    /// Add a constraint to the problem.
    pub fn add_row<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Col, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        bounds: B,
        row_factors: I,
    ) -> Row {
        let num_rows: c_int = self.num_rows().try_into().expect("too many rows");
        for r in row_factors {
            let &(col, factor) = r.borrow();
            let c = &mut self.matrix.columns[col];
            c.0.push(num_rows);
            c.1.push(factor);
        }
        self.add_row_inner(bounds)
    }

    /// Set the coefficient of a variable in a constraint.
    pub fn set_cons_coef(&mut self, row: Row, col: Col, value: f64) {
        let c = &mut self.matrix.columns[col];
        let index = c.0.iter().position(|&x| x == row as c_int);
        if let Some(index) = index {
            c.1[index] = value;
        } else {
            c.0.push(row as c_int);
            c.1.push(value);
        }
    }
}

impl From<RowMatrix> for ColMatrix {
    fn from(m: RowMatrix) -> Self {
        let mut astart = Vec::with_capacity(m.columns.len());
        astart.push(0);
        let size: usize = m.columns.iter().map(|(v, _)| v.len()).sum();
        let mut aindex = Vec::with_capacity(size);
        let mut avalue = Vec::with_capacity(size);
        for (row_indices, factors) in m.columns {
            aindex.extend_from_slice(&row_indices);
            avalue.extend_from_slice(&factors);
            astart.push(aindex.len().try_into().expect("invalid matrix size"));
        }
        Self {
            astart,
            aindex,
            avalue,
        }
    }
}

impl From<Problem<RowMatrix>> for Problem<ColMatrix> {
    fn from(pb: Problem<RowMatrix>) -> Problem<ColMatrix> {
        Self {
            colcost: pb.colcost,
            collower: pb.collower,
            colupper: pb.colupper,
            rowlower: pb.rowlower,
            rowupper: pb.rowupper,
            integrality: pb.integrality,
            matrix: pb.matrix.into(),
        }
    }
}
