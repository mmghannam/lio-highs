//! Safe rust binding to the [HiGHS](https://highs.dev) linear programming solver.
//!
//! ## Usage example
//!
//! ### Building a problem constraint by constraint with [RowProblem]
//!
//! ```
//! use lio_highs::{Sense, Model, HighsModelStatus, RowProblem, LikeModel};
//! let mut pb = RowProblem::default();
//! let x = pb.add_column(1., 0..);
//! let y = pb.add_column(2., 0..);
//! let z = pb.add_column(1., 0..);
//! pb.add_row(..=6, &[(x, 3.), (y, 1.)]);
//! pb.add_row(..=7, &[(y, 1.), (z, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```

pub mod ffi;

use ffi::*;
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_char, c_void, CStr, CString};
use std::num::TryFromIntError;
use std::ops::{Bound, Index, RangeBounds};
use std::os::raw::c_int;
use std::ptr::null_mut;

pub use matrix_col::{ColMatrix, Row};
pub use matrix_row::{Col, RowMatrix};
pub use status::{HighsModelStatus, HighsStatus};

use crate::options::HighsOptionValue;

/// A problem where variables are declared first, and constraints are then added dynamically.
pub type RowProblem = Problem<RowMatrix>;
/// A problem where constraints are declared first, and variables are then added dynamically.
pub type ColProblem = Problem<ColMatrix>;

mod matrix_col;
mod matrix_row;
mod options;
mod status;

/// A complete optimization problem.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem<MATRIX = ColMatrix> {
    // columns
    colcost: Vec<f64>,
    collower: Vec<f64>,
    colupper: Vec<f64>,
    // rows
    rowlower: Vec<f64>,
    rowupper: Vec<f64>,
    integrality: Option<Vec<HighsInt>>,
    matrix: MATRIX,
}

impl<MATRIX: Default> Problem<MATRIX>
where
    Problem<ColMatrix>: From<Problem<MATRIX>>,
{
    /// Number of variables in the problem
    pub fn num_cols(&self) -> usize {
        self.colcost.len()
    }

    /// Number of constraints in the problem
    pub fn num_rows(&self) -> usize {
        self.rowlower.len()
    }

    fn add_row_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        let r = self.num_rows().try_into().expect("too many rows");
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.rowlower.push(low);
        self.rowupper.push(high);
        r
    }

    fn add_column_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
        is_integral: bool,
    ) {
        if is_integral && self.integrality.is_none() {
            self.integrality = Some(vec![0; self.num_cols()]);
        }
        if let Some(integrality) = &mut self.integrality {
            integrality.push(if is_integral { 1 } else { 0 });
        }
        self.colcost.push(col_factor);
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.collower.push(low);
        self.colupper.push(high);
    }

    /// Create a model based on this problem. Don't solve it yet.
    pub fn optimise(self, sense: Sense) -> Model {
        self.try_optimise(sense).expect("invalid problem")
    }

    /// Create a model based on this problem. Don't solve it yet.
    pub fn try_optimise(self, sense: Sense) -> Result<Model, HighsStatus> {
        let mut m = Model::try_new(self)?;
        m.set_sense(sense);
        Ok(m)
    }

    /// Create a new problem instance
    pub fn new() -> Self {
        Self::default()
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> HighsInt {
    n.try_into().expect("size too large for HiGHS")
}

macro_rules! highs_call {
    ($function_name:ident ($($param:expr),+)) => {
        try_handle_status(
            $function_name($($param),+),
            stringify!($function_name)
        )
    }
}

/// A model to solve
#[derive(Debug, Clone)]
pub struct Model {
    highs: HighsPtr,
}

/// A solved model
#[derive(Debug, Clone)]
pub struct SolvedModel {
    highs: HighsPtr,
}

/// Whether to maximize or minimize the objective function
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// max
    Maximise = OBJECTIVE_SENSE_MAXIMIZE as isize,
    /// min
    Minimise = OBJECTIVE_SENSE_MINIMIZE as isize,
}

impl From<HighsInt> for Sense {
    fn from(value: HighsInt) -> Self {
        match value {
            OBJECTIVE_SENSE_MINIMIZE => Sense::Minimise,
            OBJECTIVE_SENSE_MAXIMIZE => Sense::Maximise,
            other => {
                panic!("Unknown objective sense with value {}", other)
            }
        }
    }
}

/// Type of a variable in the problem
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum VarType {
    /// Continuous variable (real valued)
    Continuous,
    /// Integer only variable
    Integer,
    /// Implicit Integer
    ImplicitInteger,
    /// Semi-Integer
    SemiInteger,
}

#[allow(non_upper_case_globals)]
impl From<HighsInt> for VarType {
    fn from(value: HighsInt) -> Self {
        match value {
            kHighsVarTypeContinuous => VarType::Continuous,
            kHighsVarTypeInteger => VarType::Integer,
            kHighsVarTypeImplicitInteger => VarType::ImplicitInteger,
            kHighsVarTypeSemiInteger => VarType::SemiInteger,
            other => {
                panic!("Unknown variable type with value {}", other)
            }
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new::<Problem<ColMatrix>>(Problem::default())
    }
}

impl Model {
    /// number of columns
    pub fn num_cols(&self) -> usize {
        unsafe { Highs_getNumCols(self.highs.ptr()) as usize }
    }

    /// number of rows
    pub fn num_rows(&self) -> usize {
        unsafe { Highs_getNumRows(self.highs.ptr()) as usize }
    }

    /// number of non-zeros
    pub fn num_nz(&self) -> usize {
        unsafe { Highs_getNumNz(self.highs.ptr()) as usize }
    }

    /// Returns the LP data in the problem
    pub fn get_row_lp(
        &self,
    ) -> (
        usize,
        usize,
        usize,
        Sense,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<(Col, f64)>>,
        Vec<VarType>,
    ) {
        let mut num_col = unsafe { Highs_getNumCol(self.highs.ptr()) };
        let mut num_row = unsafe { Highs_getNumRow(self.highs.ptr()) };
        let mut num_nz = unsafe { Highs_getNumNz(self.highs.ptr()) };

        let mut col_cost = vec![0.0; num_col as usize];
        let mut col_lower = vec![0.0; num_col as usize];
        let mut col_upper = vec![0.0; num_col as usize];
        let mut row_lower = vec![0.0; num_row as usize];
        let mut row_upper = vec![0.0; num_row as usize];
        let mut a_start = vec![0; num_row as usize];
        let mut a_index = vec![0; num_nz as usize];
        let mut a_value = vec![0.0; num_nz as usize];
        let mut integrality = vec![0; num_col as usize];

        let mut sense = 0i32;
        let mut offset: f64 = 0.0;

        let res = unsafe {
            highs_call! {
                Highs_getLp(
                    self.highs.ptr(),
                    kHighsMatrixFormatRowwise,
                    &mut num_col,
                    &mut num_row,
                    &mut num_nz,
                    &mut sense,
                    &mut offset,
                    col_cost.as_mut_ptr(),
                    col_lower.as_mut_ptr(),
                    col_upper.as_mut_ptr(),
                    row_lower.as_mut_ptr(),
                    row_upper.as_mut_ptr(),
                    a_start.as_mut_ptr(),
                    a_index.as_mut_ptr(),
                    a_value.as_mut_ptr(),
                    integrality.as_mut_ptr()
                )
            }
        };

        let num_col = num_col as usize;
        let num_row = num_row as usize;
        let num_nz = num_nz as usize;

        let integrality = integrality.iter().map(|v| (*v).into()).collect();

        if res.is_err() {
            panic!(
                "Failed to call Highs_getLp, got error {:?}",
                res.unwrap_err()
            );
        }

        let mut row_data = Vec::with_capacity(num_row);
        for i in 0..num_row {
            let start = a_start[i] as usize;
            let end = if i == (num_row - 1) {
                num_nz
            } else {
                a_start[i + 1] as usize
            };

            let mut index_vals = Vec::with_capacity(end - start);
            for j in start..end {
                index_vals.push((a_index[j] as Col, a_value[j]));
            }
            row_data.push(index_vals);
        }

        (
            num_col,
            num_row,
            num_nz,
            sense.into(),
            offset,
            col_cost.clone(),
            col_lower.clone(),
            col_upper.clone(),
            row_lower.clone(),
            row_upper.clone(),
            row_data.clone(),
            integrality,
        )
    }

    /// Changes the integrality of a column in the model.
    pub fn change_col_integrality(&mut self, col: Col, integer: bool) -> Result<(), HighsStatus> {
        unsafe {
            highs_call! {
                Highs_changeColIntegrality(
                    self.highs.mut_ptr(),
                    col as c_int,
                    integer as c_int
                )
            }?
        };

        Ok(())
    }

    /// Returns the LP data in the presolved problem
    pub fn get_presolved_row_lp(
        &self,
    ) -> (
        usize,
        usize,
        usize,
        Sense,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<(Col, f64)>>,
        Vec<VarType>,
    ) {
        let mut num_col = unsafe { Highs_getPresolvedNumCol(self.highs.ptr()) };
        let mut num_row = unsafe { Highs_getPresolvedNumRow(self.highs.ptr()) };
        let mut num_nz = unsafe { Highs_getPresolvedNumNz(self.highs.ptr()) };

        let mut col_cost = vec![0.0; num_col as usize];
        let mut col_lower = vec![0.0; num_col as usize];
        let mut col_upper = vec![0.0; num_col as usize];
        let mut row_lower = vec![0.0; num_row as usize];
        let mut row_upper = vec![0.0; num_row as usize];
        let mut a_start = vec![0; num_row as usize];
        let mut a_index = vec![0; num_nz as usize];
        let mut a_value = vec![0.0; num_nz as usize];
        let mut integrality = vec![0; num_col as usize];

        let mut sense = 0i32;
        let mut offset: f64 = 0.0;

        let res = unsafe {
            highs_call! {
                Highs_getPresolvedLp(
                    self.highs.ptr(),
                    kHighsMatrixFormatRowwise,
                    &mut num_col,
                    &mut num_row,
                    &mut num_nz,
                    &mut sense,
                    &mut offset,
                    col_cost.as_mut_ptr(),
                    col_lower.as_mut_ptr(),
                    col_upper.as_mut_ptr(),
                    row_lower.as_mut_ptr(),
                    row_upper.as_mut_ptr(),
                    a_start.as_mut_ptr(),
                    a_index.as_mut_ptr(),
                    a_value.as_mut_ptr(),
                    integrality.as_mut_ptr()
                )
            }
        };

        let num_col = num_col as usize;
        let num_row = num_row as usize;
        let num_nz = num_nz as usize;

        let integrality = integrality.iter().map(|v| (*v).into()).collect();

        if res.is_err() {
            panic!(
                "Failed to call Highs_getPresolvedLp, got error {:?}",
                res.unwrap_err()
            );
        }

        let mut row_data = Vec::with_capacity(num_row);
        for i in 0..num_row {
            let start = a_start[i] as usize;
            let end = if i == (num_row - 1) {
                num_nz
            } else {
                a_start[i + 1] as usize
            };

            let mut index_vals = Vec::with_capacity(end - start);
            for j in start..end {
                index_vals.push((a_index[j] as Col, a_value[j]));
            }
            row_data.push(index_vals);
        }

        (
            num_col,
            num_row,
            num_nz,
            sense.into(),
            offset,
            col_cost.clone(),
            col_lower.clone(),
            col_upper.clone(),
            row_lower.clone(),
            row_upper.clone(),
            row_data.clone(),
            integrality,
        )
    }

    /// Presolve the current model
    pub fn presolve(&mut self) {
        let ret = unsafe { Highs_presolve(self.highs.mut_ptr()) };
        assert_eq!(ret, STATUS_OK, "runPresolve failed");
    }

    /// Transform a solution from the original problem to the presolved problem.
    pub fn presolve_sol(&self, col_vals: Vec<(Col, f64)>) -> Vec<(Col, f64)> {
        let num_cols = self.num_cols();
        let num_presolved_cols = unsafe { Highs_getPresolvedNumCol(self.highs.ptr()) } as usize;

        let mut input_col_vals = vec![0.0; num_cols];
        for (col, val) in col_vals {
            if col < num_cols {
                input_col_vals[col] = val;
            }
        }

        let mut output_col_vals = vec![0.0; num_presolved_cols];

        let ret = unsafe {
            Highs_getPresolveSolution(
                self.highs.unsafe_mut_ptr(),
                input_col_vals.as_ptr(),
                output_col_vals.as_mut_ptr(),
            )
        };
        assert_eq!(ret, STATUS_OK, "presolve_sol failed");

        output_col_vals
            .into_iter()
            .enumerate()
            .map(|(col, val)| (col as Col, val))
            .collect()
    }

    /// Set the basis for the model (warm-start).
    pub fn set_basis(&mut self, col_status: &[BasisStatus], row_status: &[BasisStatus]) -> Result<(), HighsStatus> {
        assert_eq!(col_status.len(), self.num_cols(), "col_status length must match number of columns");
        assert_eq!(row_status.len(), self.num_rows(), "row_status length must match number of rows");

        let col_status_highs: Vec<HighsInt> = col_status.iter().map(|&s| s.into()).collect();
        let row_status_highs: Vec<HighsInt> = row_status.iter().map(|&s| s.into()).collect();

        unsafe {
            highs_call!(Highs_setBasis(
                self.highs.mut_ptr(),
                col_status_highs.as_ptr(),
                row_status_highs.as_ptr()
            ))?
        };

        Ok(())
    }

    /// Set the optimization sense (minimize by default)
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe { Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int) };
        assert_eq!(ret, STATUS_OK, "changeObjectiveSense failed");
    }

    /// Reads a problem
    pub fn read(&mut self, path: &str) {
        let c_path = CString::new(path).expect("invalid path");
        unsafe {
            highs_call!(Highs_readModel(self.highs.mut_ptr(), c_path.as_ptr()))
                .expect("failed to read model");
        }
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    pub fn new<P: Into<Problem<ColMatrix>>>(problem: P) -> Self {
        Self::try_new(problem).expect("incoherent problem")
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    pub fn try_new<P: Into<Problem<ColMatrix>>>(problem: P) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        let problem = problem.into();
        log::debug!(
            "Adding a problem with {} variables and {} constraints to HiGHS",
            problem.num_cols(),
            problem.num_rows()
        );
        let offset = 0.0;
        unsafe {
            if let Some(integrality) = &problem.integrality {
                highs_call!(Highs_passMip(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr(),
                    integrality.as_ptr()
                ))
            } else {
                highs_call!(Highs_passLp(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr()
                ))
            }
            .map(|_| Self { highs })
        }
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        self.highs.make_quiet()
    }

    /// Set a custom parameter on the model.
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        self.highs.set_option(option, value)
    }

    /// Find the optimal value for the problem, panic if the problem is incoherent
    pub fn solve(self) -> SolvedModel {
        self.try_solve().expect("HiGHS error: invalid problem")
    }

    /// Find the optimal value for the problem, return an error if the problem is incoherent
    pub fn try_solve(mut self) -> Result<SolvedModel, HighsStatus> {
        unsafe { highs_call!(Highs_run(self.highs.mut_ptr())) }
            .map(|_| SolvedModel { highs: self.highs })
    }

    /// Adds a new constraint to the highs model.
    pub fn add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Row {
        self.try_add_row(bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new constraint to the highs model.
    pub fn try_add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Result<Row, HighsStatus> {
        let (cols, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();

        unsafe {
            highs_call!(Highs_addRow(
                self.highs.mut_ptr(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                cols.len().try_into().unwrap(),
                cols.into_iter()
                    .map(|c| c.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(((self.highs.num_rows()? - 1) as c_int).try_into().unwrap())
    }

    /// Adds a new variable to the highs model.
    pub fn add_col(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Col {
        self.try_add_column(col_factor, bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new variable to the highs model.
    pub fn try_add_column(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Result<Col, HighsStatus> {
        let (rows, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();
        unsafe {
            highs_call!(Highs_addCol(
                self.highs.mut_ptr(),
                col_factor,
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                rows.len().try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(self.highs.num_cols()? - 1)
    }

    /// Deletes a constraint from the highs model.
    pub fn del_row(&mut self, row: Row) {
        self.try_del_row(row)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete a constraint from the highs model.
    pub fn try_del_row(&mut self, row: Row) -> Result<(), HighsStatus> {
        self.try_del_rows(vec![row])
    }

    /// Deletes constraints from the highs model.
    pub fn del_rows(&mut self, rows: Vec<Row>) {
        self.try_del_rows(rows)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete constraints from the highs model.
    pub fn try_del_rows(&mut self, rows: Vec<Row>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_deleteRowsBySet(
                self.highs.mut_ptr(),
                rows.len().try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr()
            ))?
        };

        Ok(())
    }

    /// Deletes a variable from the highs model.
    pub fn del_col(&mut self, col: Col) {
        self.try_del_col(col)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete a variable from the highs model.
    pub fn try_del_col(&mut self, col: Col) -> Result<(), HighsStatus> {
        self.try_del_cols(vec![col])
    }

    /// Deletes variables from the highs model.
    pub fn del_cols(&mut self, cols: Vec<Col>) {
        self.try_del_cols(cols)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete variables from the highs model.
    pub fn try_del_cols(&mut self, cols: Vec<Col>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_deleteColsBySet(
                self.highs.mut_ptr(),
                cols.len().try_into().unwrap(),
                cols.into_iter()
                    .map(|c| c.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr()
            ))?
        };

        Ok(())
    }

    /// Tries to change the bounds of constraints from the highs model.
    pub fn try_change_rows_bounds(
        &mut self,
        rows: Vec<Row>,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        let size = rows.len();
        unsafe {
            highs_call!(Highs_changeRowsBoundsBySet(
                self.highs.mut_ptr(),
                size.try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY); size].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY); size].as_ptr()
            ))?
        };

        Ok(())
    }

    /// Tries to change the bounds of a constraint from the highs model.
    pub fn try_change_row_bounds(
        &mut self,
        row: Row,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeRowsBoundsBySet(
                self.highs.mut_ptr(),
                1,
                vec![row as c_int].as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY)].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)].as_ptr()
            ))?
        };

        Ok(())
    }

    /// Changes the bounds of a constraint from the highs model.
    pub fn change_row_bounds(&mut self, row: Row, bounds: impl RangeBounds<f64>) {
        self.try_change_row_bounds(row, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Changes the bounds of a variable (column) in the highs model.
    pub fn change_col_bounds(&mut self, col: Col, bounds: impl RangeBounds<f64>) {
        self.try_change_col_bounds(col, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the bounds of a variable (column) in the highs model.
    pub fn try_change_col_bounds(
        &mut self,
        col: Col,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        let col_indices = [col as i32];
        unsafe {
            highs_call!(Highs_changeColsBoundsBySet(
                self.highs.mut_ptr(),
                1,
                col_indices.as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY)].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)].as_ptr()
            ))?
        };
        Ok(())
    }

    /// Changes the objective coefficient of a variable in the highs model.
    pub fn change_col_cost(&mut self, col: Col, cost: f64) {
        self.try_change_col_cost(col, cost)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the objective coefficient of a variable in the highs model.
    pub fn try_change_col_cost(&mut self, col: Col, cost: f64) -> Result<(), HighsStatus> {
        let col_indices = [col as i32];
        unsafe {
            highs_call!(Highs_changeColsCostBySet(
                self.highs.mut_ptr(),
                1,
                col_indices.as_ptr(),
                vec![cost].as_ptr()
            ))?
        };
        Ok(())
    }

    /// Check if implications data is available from MIP presolve
    pub fn has_implications(&self) -> bool {
        unsafe { Highs_hasImplications(self.highs.ptr()) != 0 }
    }

    /// Get the number of columns in the presolved model for which implications may be stored
    pub fn implications_num_col(&self) -> usize {
        unsafe { Highs_getImplicationsNumCol(self.highs.ptr()) as usize }
    }

    /// Get the number of implications for fixing a binary variable to a value.
    pub fn num_implications(&self, col: Col, val: bool) -> Result<usize, HighsStatus> {
        let mut num_implications: HighsInt = 0;
        let status = unsafe {
            Highs_getNumImplications(
                self.highs.ptr(),
                col as HighsInt,
                val as HighsInt,
                &mut num_implications,
            )
        };
        try_handle_status(status, "getting number of implications")?;
        Ok(num_implications as usize)
    }

    /// Get the implications for fixing a binary variable to a value.
    pub fn get_implications(&self, col: Col, val: bool) -> Result<Vec<Implication>, HighsStatus> {
        let num = self.num_implications(col, val)?;
        if num == 0 {
            return Ok(Vec::new());
        }

        let mut impl_cols: Vec<HighsInt> = vec![0; num];
        let mut impl_boundtypes: Vec<HighsInt> = vec![0; num];
        let mut impl_boundvals: Vec<f64> = vec![0.0; num];
        let mut num_impl: HighsInt = 0;

        let status = unsafe {
            Highs_getImplications(
                self.highs.ptr(),
                col as HighsInt,
                val as HighsInt,
                &mut num_impl,
                impl_cols.as_mut_ptr(),
                impl_boundtypes.as_mut_ptr(),
                impl_boundvals.as_mut_ptr(),
            )
        };
        try_handle_status(status, "getting implications")?;

        let implications: Vec<Implication> = (0..num_impl as usize)
            .map(|i| Implication {
                column: impl_cols[i] as Col,
                bound_type: if impl_boundtypes[i] == 0 {
                    ImplicationBoundType::Lower
                } else {
                    ImplicationBoundType::Upper
                },
                bound_value: impl_boundvals[i],
            })
            .collect();

        Ok(implications)
    }
}

impl From<SolvedModel> for Model {
    fn from(solved: SolvedModel) -> Self {
        Self {
            highs: solved.highs,
        }
    }
}

/// Logging callback that forwards HiGHS log messages to the Rust `log` crate.
unsafe extern "C" fn highs_log_callback(
    _callback_type: c_int,
    message: *const c_char,
    data_out: *const HighsCallbackDataOut,
    _data_in: *mut HighsCallbackDataIn,
    _user_data: *mut c_void,
) {
    if message.is_null() || data_out.is_null() {
        return;
    }
    let msg = CStr::from_ptr(message).to_string_lossy();
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    let log_type = (*data_out).log_type;
    match log_type {
        4 => log::warn!("HiGHS: {}", msg),
        5 => log::error!("HiGHS: {}", msg),
        _ => log::trace!("HiGHS: {}", msg),
    }
}

/// Wrapper around a HiGHS pointer.
#[derive(Debug, Clone)]
pub struct HighsPtr(*mut c_void);

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl Default for HighsPtr {
    fn default() -> Self {
        let ptr = unsafe { Highs_create() };
        unsafe {
            Highs_setCallback(ptr, Some(highs_log_callback), null_mut());
            Highs_startCallback(ptr, kHighsCallbackLogging);
        }
        Self(ptr)
    }
}

/// Trait to give access to methods that are common to both `Model` and `SolvedModel`.
pub trait LikeModel {
    /// Returns the number of columns in the model
    fn num_cols(&self) -> usize;

    /// Returns the name of the variable at the given index.
    fn get_col_name(&self, col: Col) -> Result<String, HighsStatus>;

    /// The status of the solution. Should be Optimal if everything went well
    fn status(&self) -> HighsModelStatus;

    /// Get data associated with multiple adjacent rows from the model
    fn get_rows_by_range(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            usize,
            Vec<HighsInt>,
            Vec<HighsInt>,
            Vec<f64>,
        ),
        HighsStatus,
    >;

    /// Get rows by range in a more structured format
    fn get_rows_by_range_structured(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<Vec<RowData>, HighsStatus> {
        let (lower, upper, _num_nz, matrix_start, matrix_index, matrix_value) =
            self.get_rows_by_range(from_row, to_row)?;

        let num_rows = (to_row - from_row + 1) as usize;
        let mut rows = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let start = matrix_start[i] as usize;
            let end = if i == num_rows - 1 {
                matrix_value.len()
            } else {
                matrix_start[i + 1] as usize
            };

            let mut coefficients = Vec::with_capacity(end - start);
            for j in start..end {
                coefficients.push((matrix_index[j] as Col, matrix_value[j]));
            }

            rows.push(RowData {
                lower_bound: lower[i],
                upper_bound: upper[i],
                coefficients,
            });
        }

        Ok(rows)
    }

    /// Get data for a single row/constraint
    fn get_row(&self, row: Row) -> Result<RowData, HighsStatus> {
        let mut row_data = self.get_rows_by_range_structured(row, row)?;
        Ok(row_data.pop().unwrap())
    }

    /// Writes the model to a file
    fn write(&self, path: &str) -> Result<(), HighsStatus>;

    /// Postsolve a solution
    fn postsolve(&mut self, col_vals: Vec<(Col, f64)>) -> Vec<(Col, f64)>;

    /// Get the solution
    fn get_solution(&self) -> Solution;
}

impl<H: HasHighsPtr> LikeModel for H {
    fn postsolve(&mut self, col_vals: Vec<(Col, f64)>) -> Vec<(Col, f64)> {
        let mut col_vals = col_vals;
        let num_cols = self.num_cols();
        let mut flat_col_vals = vec![0.0; num_cols];
        for (col, val) in col_vals.iter_mut() {
            flat_col_vals[*col] = *val;
        }
        let col_vals_ptr = flat_col_vals.as_mut_ptr();
        let ret = unsafe {
            Highs_postsolve(
                self.highs_ptr().0,
                col_vals_ptr as *const f64,
                null_mut(),
                null_mut(),
            )
        };
        assert_ne!(ret, STATUS_ERROR, "postsolve failed");

        let solution = self.get_solution();
        for (col, val) in flat_col_vals.iter_mut().enumerate() {
            *val = solution.colvalue[col];
        }

        flat_col_vals
            .into_iter()
            .enumerate()
            .map(|(col, val)| (col as Col, val))
            .collect()
    }

    fn get_col_name(&self, col: Col) -> Result<String, HighsStatus> {
        let highs_ptr = self.highs_ptr();
        let mut name_buf = vec![0u8; kHighsMaximumStringLength as usize];
        unsafe {
            highs_call!(Highs_getColName(
                highs_ptr.unsafe_mut_ptr(),
                col as i32,
                name_buf.as_mut_ptr() as *mut i8
            ))?;
        }

        let name = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as char)
            .collect::<String>();

        Ok(name)
    }

    fn status(&self) -> HighsModelStatus {
        let model_status = unsafe { Highs_getModelStatus(self.highs_ptr().unsafe_mut_ptr()) };
        HighsModelStatus::try_from(model_status).unwrap()
    }

    fn get_rows_by_range(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            usize,
            Vec<HighsInt>,
            Vec<HighsInt>,
            Vec<f64>,
        ),
        HighsStatus,
    > {
        let num_rows = (to_row - from_row + 1) as usize;

        let mut lower = vec![0.0; num_rows];
        let mut upper = vec![0.0; num_rows];
        let mut num_nz = 0i32;

        unsafe {
            highs_call!(Highs_getRowsByRange(
                self.highs_ptr().unsafe_mut_ptr(),
                from_row as c_int,
                to_row as c_int,
                &mut (num_rows as c_int),
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
                &mut num_nz,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut()
            ))?;
        }

        let mut matrix_start = vec![0; num_rows];
        let mut matrix_index = vec![0; num_nz as usize];
        let mut matrix_value = vec![0.0; num_nz as usize];
        let mut num_rows_out = num_rows as c_int;

        unsafe {
            highs_call!(Highs_getRowsByRange(
                self.highs_ptr().unsafe_mut_ptr(),
                from_row as c_int,
                to_row as c_int,
                &mut num_rows_out,
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
                &mut num_nz,
                matrix_start.as_mut_ptr(),
                matrix_index.as_mut_ptr(),
                matrix_value.as_mut_ptr()
            ))?;
        }

        Ok((
            lower,
            upper,
            num_nz as usize,
            matrix_start,
            matrix_index,
            matrix_value,
        ))
    }

    fn write(&self, path: &str) -> Result<(), HighsStatus> {
        let c_path = CString::new(path).expect("invalid path");
        unsafe {
            highs_call!(Highs_writeModel(
                self.highs_ptr().unsafe_mut_ptr(),
                c_path.as_ptr()
            ))
        }?;

        Ok(())
    }

    fn num_cols(&self) -> usize {
        self.highs_ptr().num_cols().unwrap()
    }

    fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.highs_ptr().num_rows().unwrap();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        unsafe {
            Highs_getSolution(
                self.highs_ptr().unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }
}

impl HighsPtr {
    const fn ptr(&self) -> *const c_void {
        self.0
    }

    unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        self.set_option(&b"output_flag"[..], false);
        self.set_option(&b"log_to_console"[..], false);
    }

    /// Set a custom parameter on the model
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        let c_str = CString::new(option).expect("invalid option name");
        let status = unsafe { value.apply_to_highs(self.mut_ptr(), c_str.as_ptr()) };
        try_handle_status(status, "Highs_setOptionValue")
            .expect("An error was encountered in HiGHS.");
    }

    /// Number of variables
    pub fn num_cols(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumCols(self.0) };
        n.try_into()
    }

    /// Number of constraints
    pub fn num_rows(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumRows(self.0) };
        n.try_into()
    }
}

impl SolvedModel {
    /// Get the solution to the problem
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Number of variables
    pub fn num_cols(&self) -> usize {
        self.highs.num_cols().expect("invalid number of columns")
    }

    /// Number of constraints
    pub fn num_rows(&self) -> usize {
        self.highs.num_rows().expect("invalid number of rows")
    }

    /// Get the basis variables
    pub fn get_basic_vars(&self) -> Vec<BasicVar> {
        let mut basis_ids = vec![0; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasicVariables(self.highs.unsafe_mut_ptr(), basis_ids.as_mut_ptr())
            }
            .map_err(|e| {
                println!("Error while getting basic variables: {:?}", e);
            })
            .unwrap();
        }

        let mut res = Vec::with_capacity(self.num_rows());

        for basis_var in basis_ids.into_iter() {
            if basis_var >= 0 {
                res.push(BasicVar::Col(basis_var as Col));
            } else {
                res.push(BasicVar::Row((-basis_var - 1) as Row));
            }
        }

        res
    }

    /// Get basis status
    pub fn get_basis_status(&self) -> (Vec<BasisStatus>, Vec<BasisStatus>) {
        let mut col_status = vec![kHighsBasisStatusZero; self.num_cols()];
        let mut row_status = vec![kHighsBasisStatusZero; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasis(
                    self.highs.unsafe_mut_ptr(),
                    col_status.as_mut_ptr(),
                    row_status.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis status: {:?}", e);
            })
            .unwrap();
        }

        let col_status = col_status.iter().map(|&s| s.into()).collect();
        let row_status = row_status.iter().map(|&s| s.into()).collect();
        (col_status, row_status)
    }

    /// Get the reduced row
    pub fn get_reduced_row(&self, row: Row) -> (Vec<f64>, Vec<HighsInt>) {
        let mut reduced_row = vec![0.; self.num_cols()];
        let row_non_zeros: *mut HighsInt = &mut 0;
        let mut row_index: Vec<HighsInt> = vec![0; self.num_cols()];
        unsafe {
            highs_call! {
                Highs_getReducedRow(
                    self.highs.unsafe_mut_ptr(),
                    row.try_into().unwrap(),
                    reduced_row.as_mut_ptr(),
                    row_non_zeros,
                    row_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting reduced row: {:?}", e);
            })
            .unwrap();
        }
        let num_nonzeros = unsafe { *row_non_zeros };
        row_index = row_index.into_iter().take(num_nonzeros as usize).collect();

        (reduced_row, row_index)
    }

    /// Get the reduced column
    pub fn get_reduced_column(&self, col: Col) -> (Vec<f64>, Vec<HighsInt>) {
        let mut reduced_col = vec![0.; self.num_rows()];
        let col_non_zeros: *mut HighsInt = &mut 0;
        let mut col_index = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getReducedColumn(
                    self.highs.unsafe_mut_ptr(),
                    col.try_into().unwrap(),
                    reduced_col.as_mut_ptr(),
                    col_non_zeros,
                    col_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting reduced column: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *col_non_zeros };
        col_index = col_index.into_iter().take(num_nonzeros as usize).collect();

        (reduced_col, col_index)
    }

    /// Returns solution to x = B^{-1} * b
    pub fn get_basis_sol(&self, mut b: Vec<f64>) -> (Vec<f64>, Vec<HighsInt>) {
        let mut x = vec![0.; self.num_rows()];
        let solution_num_nz: *mut HighsInt = &mut 0;
        let mut solution_index: Vec<HighsInt> = vec![0; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasisSolve(
                    self.highs.unsafe_mut_ptr(),
                    b.as_mut_ptr(),
                    x.as_mut_ptr(),
                    solution_num_nz,
                    solution_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse row: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *solution_num_nz };
        solution_index = solution_index
            .into_iter()
            .take(num_nonzeros as usize)
            .collect();

        (x, solution_index)
    }

    /// Gets a row of the basis inverse matrix B^{-1}
    pub fn get_basis_inverse_row(&self, row: Row) -> (Vec<f64>, Vec<HighsInt>) {
        let mut row_vector = vec![0.; self.num_rows()];
        let row_num_nz: *mut HighsInt = &mut 0;
        let mut row_index: Vec<HighsInt> = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getBasisInverseRow(
                    self.highs.unsafe_mut_ptr(),
                    row.try_into().unwrap(),
                    row_vector.as_mut_ptr(),
                    row_num_nz,
                    row_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse row: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *row_num_nz };
        row_index = row_index.into_iter().take(num_nonzeros as usize).collect();

        (row_vector, row_index)
    }

    /// Gets a column of the basis inverse matrix B^{-1}
    pub fn get_basis_inverse_col(&self, col: Col) -> (Vec<f64>, Vec<HighsInt>) {
        let mut col_vector = vec![0.; self.num_rows()];
        let col_num_nz: *mut HighsInt = &mut 0;
        let mut col_index = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getBasisInverseCol(
                    self.highs.unsafe_mut_ptr(),
                    col.try_into().unwrap(),
                    col_vector.as_mut_ptr(),
                    col_num_nz,
                    col_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse column: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *col_num_nz };
        col_index = col_index.into_iter().take(num_nonzeros as usize).collect();

        (col_vector, col_index)
    }

    /// Gets the dual ray (Farkas proof) for an infeasible LP.
    pub fn get_dual_ray(&self) -> Option<Vec<f64>> {
        let mut has_dual_ray: HighsInt = 0;
        let mut dual_ray_value: Vec<f64> = vec![0.; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getDualRay(
                    self.highs.unsafe_mut_ptr(),
                    &mut has_dual_ray,
                    dual_ray_value.as_mut_ptr()
                )
            }
            .map_err(|e| {
                eprintln!("Error while getting dual ray: {:?}", e);
            })
            .ok()?;
        }

        if has_dual_ray != 0 {
            Some(dual_ray_value)
        } else {
            None
        }
    }

    /// Gets the objective value
    pub fn obj_val(&self) -> f64 {
        unsafe { Highs_getObjectiveValue(self.highs.ptr()) }
    }

    /// Gets the total iteration count
    pub fn get_iteration_count(&self) -> i32 {
        unsafe { Highs_getIterationCount(self.highs.ptr()) }
    }

    /// Gets the simplex iteration count
    pub fn get_simplex_iteration_count(&self) -> i32 {
        let mut value: HighsInt = 0;
        let info_name = CString::new("simplex_iteration_count").unwrap();
        unsafe {
            Highs_getIntInfoValue(self.highs.ptr(), info_name.as_ptr(), &mut value);
        }
        value
    }

    /// Check if implications data is available from MIP presolve
    pub fn has_implications(&self) -> bool {
        unsafe { Highs_hasImplications(self.highs.ptr()) != 0 }
    }

    /// Get the number of columns in the presolved model for which implications may be stored
    pub fn implications_num_col(&self) -> usize {
        unsafe { Highs_getImplicationsNumCol(self.highs.ptr()) as usize }
    }

    /// Get the number of implications for fixing a binary variable to a value.
    pub fn num_implications(&self, col: Col, val: bool) -> Result<usize, HighsStatus> {
        let mut num_implications: HighsInt = 0;
        let status = unsafe {
            Highs_getNumImplications(
                self.highs.ptr(),
                col as HighsInt,
                val as HighsInt,
                &mut num_implications,
            )
        };
        try_handle_status(status, "getting number of implications")?;
        Ok(num_implications as usize)
    }

    /// Get the implications for fixing a binary variable to a value.
    pub fn get_implications(&self, col: Col, val: bool) -> Result<Vec<Implication>, HighsStatus> {
        let num = self.num_implications(col, val)?;
        if num == 0 {
            return Ok(Vec::new());
        }

        let mut impl_cols: Vec<HighsInt> = vec![0; num];
        let mut impl_boundtypes: Vec<HighsInt> = vec![0; num];
        let mut impl_boundvals: Vec<f64> = vec![0.0; num];
        let mut num_impl: HighsInt = 0;

        let status = unsafe {
            Highs_getImplications(
                self.highs.ptr(),
                col as HighsInt,
                val as HighsInt,
                &mut num_impl,
                impl_cols.as_mut_ptr(),
                impl_boundtypes.as_mut_ptr(),
                impl_boundvals.as_mut_ptr(),
            )
        };
        try_handle_status(status, "getting implications")?;

        let implications: Vec<Implication> = (0..num_impl as usize)
            .map(|i| Implication {
                column: impl_cols[i] as Col,
                bound_type: if impl_boundtypes[i] == 0 {
                    ImplicationBoundType::Lower
                } else {
                    ImplicationBoundType::Upper
                },
                bound_value: impl_boundvals[i],
            })
            .collect();

        Ok(implications)
    }
}

/// Trait for types that can provide access to the underlying HiGHS pointer
pub trait HasHighsPtr {
    /// Get the underlying HiGHS pointer
    fn highs_ptr(&self) -> &HighsPtr;

    /// Get a mutable reference to the underlying HiGHS pointer
    fn highs_mut_ptr(&mut self) -> &mut HighsPtr;
}

impl HasHighsPtr for Model {
    fn highs_ptr(&self) -> &HighsPtr {
        &self.highs
    }

    fn highs_mut_ptr(&mut self) -> &mut HighsPtr {
        &mut self.highs
    }
}

impl HasHighsPtr for SolvedModel {
    fn highs_ptr(&self) -> &HighsPtr {
        &self.highs
    }

    fn highs_mut_ptr(&mut self) -> &mut HighsPtr {
        &mut self.highs
    }
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    colvalue: Vec<f64>,
    coldual: Vec<f64>,
    rowvalue: Vec<f64>,
    rowdual: Vec<f64>,
}

/// Data for a single row/constraint
#[derive(Clone, Debug)]
pub struct RowData {
    /// Lower bound of the constraint
    pub lower_bound: f64,
    /// Upper bound of the constraint
    pub upper_bound: f64,
    /// Constraint coefficients as (column_index, coefficient) pairs
    pub coefficients: Vec<(Col, f64)>,
}

impl Solution {
    /// The optimal values for each variables (in the order they were added)
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    /// The optimal values for each variables in the dual problem (in the order they were added)
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    /// The value of the constraint functions
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
    /// The value of the constraint functions in the dual problem
    pub fn dual_rows(&self) -> &[f64] {
        &self.rowdual
    }
}

impl Index<Col> for Solution {
    type Output = f64;
    fn index(&self, col: Col) -> &f64 {
        &self.colvalue[col]
    }
}

fn try_handle_status(status: c_int, msg: &str) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status)
        .expect("HiGHS returned an unexpected status value. Please report it as a bug.");
    match status_enum {
        status @ HighsStatus::OK => Ok(status),
        status @ HighsStatus::Warning => {
            log::debug!("HiGHS returned warning status from {}", msg);
            Ok(status)
        }
        error => Err(error),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// The status of a variable/column in the basis
pub enum BasisStatus {
    /// The variable is at its lower bound
    Lower,
    /// The variable is basic
    Basic,
    /// The variable is at its upper bound
    Upper,
    /// The variable is at zero
    Zero,
}

/// A basic variable is either a column or a row
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicVar {
    /// Basic variable
    Col(Col),
    /// Basic row
    Row(Row),
}

/// The type of bound in an implication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplicationBoundType {
    /// Lower bound implication
    Lower,
    /// Upper bound implication
    Upper,
}

/// An implication discovered during MIP presolve.
#[derive(Debug, Clone, PartialEq)]
pub struct Implication {
    /// The column index (in the presolved model) that is affected
    pub column: Col,
    /// The type of bound (lower or upper)
    pub bound_type: ImplicationBoundType,
    /// The bound value
    pub bound_value: f64,
}

#[allow(non_upper_case_globals)]
impl From<HighsInt> for BasisStatus {
    fn from(status: HighsInt) -> Self {
        match status {
            kHighsBasisStatusLower => BasisStatus::Lower,
            kHighsBasisStatusBasic => BasisStatus::Basic,
            kHighsBasisStatusUpper => BasisStatus::Upper,
            kHighsBasisStatusZero => BasisStatus::Zero,
            _ => panic!("Invalid basis status"),
        }
    }
}

#[allow(non_upper_case_globals)]
impl From<BasisStatus> for HighsInt {
    fn from(status: BasisStatus) -> Self {
        match status {
            BasisStatus::Lower => kHighsBasisStatusLower,
            BasisStatus::Basic => kHighsBasisStatusBasic,
            BasisStatus::Upper => kHighsBasisStatusUpper,
            BasisStatus::Zero => kHighsBasisStatusZero,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_lp() {
        let mut pb = RowProblem::default();
        let x = pb.add_column(1., 0..);
        let y = pb.add_column(2., 0..);
        let z = pb.add_column(1., 0..);
        pb.add_row(..=6, &[(x, 3.), (y, 1.)]);
        pb.add_row(..=7, &[(y, 1.), (z, 2.)]);
        let solved = pb.optimise(Sense::Maximise).solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_eq!(solution.columns(), vec![0., 6., 0.5]);
        assert_eq!(solution.rows(), vec![6., 7.]);
    }
}
