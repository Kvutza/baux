//! Physics specification and state management for MuJoCo control benchmarks.
//! Linked directly against the official C `libmujoco` physics engine via
//! `mujoco_shim.c`.

use baux::hash::splitmix64;
use std::ffi::c_void;
use std::sync::Arc;

unsafe extern "C" {
    fn mjshim_load(xml: *const u8, len: usize, err: *mut u8, err_sz: i32) -> *mut c_void;
    fn mjshim_cheetah(err: *mut u8, err_sz: i32) -> *mut c_void;
    fn mjshim_hopper(err: *mut u8, err_sz: i32) -> *mut c_void;
    fn mjshim_free(m: *mut c_void);
    fn mjshim_data(m: *const c_void) -> *mut c_void;
    fn mjshim_drop(d: *mut c_void);
    fn mjshim_reset(m: *const c_void, d: *mut c_void);
    fn mjshim_forward(m: *const c_void, d: *mut c_void);
    fn mjshim_step(m: *const c_void, d: *mut c_void);
    fn mjshim_nq(m: *const c_void) -> i32;
    fn mjshim_nv(m: *const c_void) -> i32;
    fn mjshim_nu(m: *const c_void) -> i32;
    fn mjshim_timestep(m: *const c_void) -> f64;
    fn mjshim_qpos0(m: *const c_void) -> *const f64;
    fn mjshim_qpos(d: *mut c_void) -> *mut f64;
    fn mjshim_qvel(d: *mut c_void) -> *mut f64;
    fn mjshim_ctrl(d: *mut c_void) -> *mut f64;
    fn mjshim_time(d: *const c_void) -> f64;
}

#[derive(Debug)]
struct MjModelHandle(*mut c_void);

unsafe impl Send for MjModelHandle {}
unsafe impl Sync for MjModelHandle {}

impl Drop for MjModelHandle {
    fn drop(&mut self) {
        unsafe {
            mjshim_free(self.0);
        }
    }
}

/// Static configuration and compiled C `mjModel` describing a simulated rigid
/// body system.
#[derive(Clone, Debug)]
pub struct MjModelSpec {
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub dt: f32,
    pub frame_skip: usize,
    pub max_episode_steps: usize,
    pub reset_noise_scale: f32,
    pub qvel_gaussian: bool,
    pub qpos0: Vec<f32>,
    pub ctrl_low: Vec<f32>,
    pub ctrl_high: Vec<f32>,
    handle: Arc<MjModelHandle>,
}

impl PartialEq for MjModelSpec {
    fn eq(&self, other: &Self) -> bool {
        self.nq == other.nq
            && self.nv == other.nv
            && self.nu == other.nu
            && self.dt == other.dt
            && self.frame_skip == other.frame_skip
            && self.max_episode_steps == other.max_episode_steps
            && self.reset_noise_scale == other.reset_noise_scale
            && self.qvel_gaussian == other.qvel_gaussian
            && self.qpos0 == other.qpos0
            && self.ctrl_low == other.ctrl_low
            && self.ctrl_high == other.ctrl_high
    }
}

impl MjModelSpec {
    fn from_ptr(
        ptr: *mut c_void,
        err_buf: &[u8; 512],
        frame_skip: usize,
        max_episode_steps: usize,
        reset_noise_scale: f32,
        qvel_gaussian: bool,
    ) -> Self {
        if ptr.is_null() {
            let msg = String::from_utf8_lossy(err_buf);
            panic!("failed to load MuJoCo XML: {}", msg.trim_end_matches('\0'));
        }
        let nq = unsafe { mjshim_nq(ptr) as usize };
        let nv = unsafe { mjshim_nv(ptr) as usize };
        let nu = unsafe { mjshim_nu(ptr) as usize };
        let dt = unsafe { mjshim_timestep(ptr) as f32 };
        let qpos0_ptr = unsafe { mjshim_qpos0(ptr) };
        let qpos0_slice = unsafe { std::slice::from_raw_parts(qpos0_ptr, nq) };
        let qpos0 = qpos0_slice.iter().map(|&x| x as f32).collect();
        Self {
            nq,
            nv,
            nu,
            dt,
            frame_skip,
            max_episode_steps,
            reset_noise_scale,
            qvel_gaussian,
            qpos0,
            ctrl_low: vec![-1.0; nu],
            ctrl_high: vec![1.0; nu],
            handle: Arc::new(MjModelHandle(ptr)),
        }
    }

    pub fn from_xml(
        xml: &str,
        frame_skip: usize,
        max_episode_steps: usize,
        reset_noise_scale: f32,
        qvel_gaussian: bool,
    ) -> Self {
        let mut err_buf = [0u8; 512];
        let ptr = unsafe {
            mjshim_load(
                xml.as_ptr(),
                xml.len(),
                err_buf.as_mut_ptr(),
                err_buf.len() as i32,
            )
        };
        Self::from_ptr(
            ptr,
            &err_buf,
            frame_skip,
            max_episode_steps,
            reset_noise_scale,
            qvel_gaussian,
        )
    }

    pub fn cheetah() -> Self {
        let mut err_buf = [0u8; 512];
        let ptr = unsafe { mjshim_cheetah(err_buf.as_mut_ptr(), err_buf.len() as i32) };
        Self::from_ptr(ptr, &err_buf, 5, 1000, 0.1, true)
    }

    pub fn hopper() -> Self {
        let mut err_buf = [0u8; 512];
        let ptr = unsafe { mjshim_hopper(err_buf.as_mut_ptr(), err_buf.len() as i32) };
        Self::from_ptr(ptr, &err_buf, 4, 1000, 0.005, false)
    }

    fn raw(&self) -> *const c_void {
        self.handle.0
    }
}

/// Dynamic state and allocated C `mjData` buffers for an active MuJoCo rollout.
#[derive(Debug)]
pub struct MjDataState {
    pub qpos: Vec<f32>,
    pub qvel: Vec<f32>,
    pub ctrl: Vec<f32>,
    pub time: f32,
    pub steps: usize,
    model_handle: Arc<MjModelHandle>,
    data_ptr: *mut c_void,
}

unsafe impl Send for MjDataState {}

impl Drop for MjDataState {
    fn drop(&mut self) {
        unsafe {
            mjshim_drop(self.data_ptr);
        }
    }
}

impl Clone for MjDataState {
    fn clone(&self) -> Self {
        let data_ptr = unsafe { mjshim_data(self.model_handle.0) };
        assert!(!data_ptr.is_null(), "mj_makeData failed");
        let mut cloned = Self {
            qpos: self.qpos.clone(),
            qvel: self.qvel.clone(),
            ctrl: self.ctrl.clone(),
            time: self.time,
            steps: self.steps,
            model_handle: Arc::clone(&self.model_handle),
            data_ptr,
        };
        cloned.write_c();
        cloned
    }
}

impl PartialEq for MjDataState {
    fn eq(&self, other: &Self) -> bool {
        self.qpos == other.qpos
            && self.qvel == other.qvel
            && self.ctrl == other.ctrl
            && self.time == other.time
            && self.steps == other.steps
    }
}

fn sample_uniform(prng: &mut u64) -> f64 {
    *prng = splitmix64(*prng);
    (*prng as f64) / (u64::MAX as f64)
}

fn sample_normal(prng: &mut u64) -> f64 {
    let u1 = sample_uniform(prng).clamp(1e-12, 1.0 - 1e-12);
    let u2 = sample_uniform(prng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl MjDataState {
    /// Constructs a state initialized to match the compiled `mjModel`.
    pub fn new(model: &MjModelSpec) -> Self {
        let data_ptr = unsafe { mjshim_data(model.raw()) };
        assert!(!data_ptr.is_null(), "mj_makeData failed");
        let mut state = Self {
            qpos: model.qpos0.clone(),
            qvel: vec![0.0; model.nv],
            ctrl: vec![0.0; model.nu],
            time: 0.0,
            steps: 0,
            model_handle: Arc::clone(&model.handle),
            data_ptr,
        };
        state.sync_c();
        state
    }

    fn sync_c(&mut self) {
        unsafe {
            let qp = std::slice::from_raw_parts(mjshim_qpos(self.data_ptr), self.qpos.len());
            let qv = std::slice::from_raw_parts(mjshim_qvel(self.data_ptr), self.qvel.len());
            let ct = std::slice::from_raw_parts(mjshim_ctrl(self.data_ptr), self.ctrl.len());
            for (dst, &src) in self.qpos.iter_mut().zip(qp.iter()) {
                *dst = src as f32;
            }
            for (dst, &src) in self.qvel.iter_mut().zip(qv.iter()) {
                *dst = src as f32;
            }
            for (dst, &src) in self.ctrl.iter_mut().zip(ct.iter()) {
                *dst = src as f32;
            }
            self.time = mjshim_time(self.data_ptr) as f32;
        }
    }

    fn write_c(&mut self) {
        unsafe {
            let qp = std::slice::from_raw_parts_mut(mjshim_qpos(self.data_ptr), self.qpos.len());
            let qv = std::slice::from_raw_parts_mut(mjshim_qvel(self.data_ptr), self.qvel.len());
            let ct = std::slice::from_raw_parts_mut(mjshim_ctrl(self.data_ptr), self.ctrl.len());
            for (dst, &src) in qp.iter_mut().zip(self.qpos.iter()) {
                *dst = f64::from(src);
            }
            for (dst, &src) in qv.iter_mut().zip(self.qvel.iter()) {
                *dst = f64::from(src);
            }
            for (dst, &src) in ct.iter_mut().zip(self.ctrl.iter()) {
                *dst = f64::from(src);
            }
            mjshim_forward(self.model_handle.0, self.data_ptr);
        }
        self.sync_c();
    }

    /// Resets dynamic `mjData` state in-place and runs `mj_forward`.
    pub fn reset_from(&mut self, model: &MjModelSpec, seed: u64) {
        self.steps = 0;
        let mut prng = seed;
        let scale = f64::from(model.reset_noise_scale);
        unsafe {
            mjshim_reset(model.raw(), self.data_ptr);
            let qp = std::slice::from_raw_parts_mut(mjshim_qpos(self.data_ptr), model.nq);
            let qv = std::slice::from_raw_parts_mut(mjshim_qvel(self.data_ptr), model.nv);
            for i in 0..model.nq {
                let u = sample_uniform(&mut prng);
                let noise = (u * 2.0 - 1.0) * scale;
                qp[i] = f64::from(model.qpos0[i]) + noise;
            }
            for i in 0..model.nv {
                let noise = if model.qvel_gaussian {
                    sample_normal(&mut prng) * scale
                } else {
                    (sample_uniform(&mut prng) * 2.0 - 1.0) * scale
                };
                qv[i] = noise;
            }
            mjshim_forward(model.raw(), self.data_ptr);
        }
        self.sync_c();
    }

    /// Sets actuators mapped to control range in both Rust and `mjData`.
    pub fn set_ctrl(&mut self, model: &MjModelSpec, action: &[f32]) {
        unsafe {
            let ct = std::slice::from_raw_parts_mut(mjshim_ctrl(self.data_ptr), model.nu);
            for i in 0..model.nu {
                let a = if i < action.len() {
                    action[i].clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let low = model.ctrl_low[i];
                let high = model.ctrl_high[i];
                let mapped = (low + (a + 1.0) * 0.5 * (high - low)).clamp(low, high);
                self.ctrl[i] = mapped;
                ct[i] = f64::from(mapped);
            }
        }
    }

    /// Advances the C `mj_step` simulation by `frame_skip` sub-steps.
    pub fn step_c(&mut self, model: &MjModelSpec) {
        unsafe {
            for _ in 0..model.frame_skip {
                mjshim_step(model.raw(), self.data_ptr);
            }
        }
        self.sync_c();
    }
}

/// Outcome of advancing the task forward by one decision macro-step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub reward_total: f32,
    pub reward_forward: f32,
    pub reward_ctrl: f32,
    pub terminated: bool,
    pub truncated: bool,
}

/// Abstract contract for a deterministic continuous control benchmark.
pub trait ControlTask {
    fn model(&self) -> &MjModelSpec;
    fn obs_dim(&self) -> usize;
    fn reset(&self, data: &mut MjDataState, seed: u64);
    fn extract_obs(&self, data: &MjDataState, obs_out: &mut [f32]);
    fn step(&self, data: &mut MjDataState, action: &[f32]) -> StepOutcome;
}

/// Official MuJoCo 2D planar bipedal `HalfCheetah-v5` benchmark.
#[derive(Clone, Debug)]
pub struct HalfCheetahTask {
    model: MjModelSpec,
    forward_reward_weight: f32,
    ctrl_cost_weight: f32,
}

impl HalfCheetahTask {
    pub fn new() -> Self {
        let model = MjModelSpec::cheetah();
        Self {
            model,
            forward_reward_weight: 1.0,
            ctrl_cost_weight: 0.1,
        }
    }
}

impl Default for HalfCheetahTask {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlTask for HalfCheetahTask {
    fn model(&self) -> &MjModelSpec {
        &self.model
    }

    fn obs_dim(&self) -> usize {
        17
    }

    fn reset(&self, data: &mut MjDataState, seed: u64) {
        data.reset_from(&self.model, seed);
    }

    fn extract_obs(&self, data: &MjDataState, obs_out: &mut [f32]) {
        obs_out[..8].copy_from_slice(&data.qpos[1..9]);
        obs_out[8..17].copy_from_slice(&data.qvel[..9]);
    }

    fn step(&self, data: &mut MjDataState, action: &[f32]) -> StepOutcome {
        data.set_ctrl(&self.model, action);
        let x_before = unsafe { *mjshim_qpos(data.data_ptr) };
        data.step_c(&self.model);
        data.steps += 1;
        let x_after = unsafe { *mjshim_qpos(data.data_ptr) };
        let total_dt = f64::from(self.model.dt) * (self.model.frame_skip as f64);
        let forward_vel = ((x_after - x_before) / total_dt) as f32;
        let reward_fwd = self.forward_reward_weight * forward_vel;
        let ctrl_cost = self.ctrl_cost_weight * data.ctrl.iter().map(|&u| u * u).sum::<f32>();
        let reward_total = reward_fwd - ctrl_cost;
        let truncated = data.steps >= self.model.max_episode_steps;
        StepOutcome {
            reward_total,
            reward_forward: reward_fwd,
            reward_ctrl: ctrl_cost,
            terminated: false,
            truncated,
        }
    }
}

/// Official MuJoCo 2D planar single-legged `Hopper-v5` benchmark.
#[derive(Clone, Debug)]
pub struct HopperTask {
    model: MjModelSpec,
    forward_reward_weight: f32,
    ctrl_cost_weight: f32,
    healthy_reward: f32,
}

impl HopperTask {
    pub fn new() -> Self {
        let model = MjModelSpec::hopper();
        Self {
            model,
            forward_reward_weight: 1.0,
            ctrl_cost_weight: 0.001,
            healthy_reward: 1.0,
        }
    }
}

impl Default for HopperTask {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlTask for HopperTask {
    fn model(&self) -> &MjModelSpec {
        &self.model
    }

    fn obs_dim(&self) -> usize {
        11
    }

    fn reset(&self, data: &mut MjDataState, seed: u64) {
        data.reset_from(&self.model, seed);
    }

    fn extract_obs(&self, data: &MjDataState, obs_out: &mut [f32]) {
        obs_out[..5].copy_from_slice(&data.qpos[1..6]);
        for i in 0..6 {
            obs_out[5 + i] = data.qvel[i].clamp(-10.0, 10.0);
        }
    }

    fn step(&self, data: &mut MjDataState, action: &[f32]) -> StepOutcome {
        data.set_ctrl(&self.model, action);
        let x_before = unsafe { *mjshim_qpos(data.data_ptr) };
        data.step_c(&self.model);
        data.steps += 1;
        let x_after = unsafe { *mjshim_qpos(data.data_ptr) };
        let total_dt = f64::from(self.model.dt) * (self.model.frame_skip as f64);
        let forward_vel = ((x_after - x_before) / total_dt) as f32;
        let reward_fwd = self.forward_reward_weight * forward_vel;
        let ctrl_cost = self.ctrl_cost_weight * data.ctrl.iter().map(|&u| u * u).sum::<f32>();
        let z = data.qpos[1];
        let angle = data.qpos[2];
        let state_healthy = data.qpos[2..]
            .iter()
            .chain(data.qvel.iter())
            .all(|&v| v.is_finite() && (-100.0..=100.0).contains(&v));
        let is_healthy = state_healthy && z > 0.7 && angle.abs() < 0.2;
        let terminated = !is_healthy;
        let truncated = data.steps >= self.model.max_episode_steps;
        let reward_total = if is_healthy {
            reward_fwd + self.healthy_reward - ctrl_cost
        } else {
            reward_fwd - ctrl_cost
        };
        StepOutcome {
            reward_total,
            reward_forward: reward_fwd,
            reward_ctrl: ctrl_cost,
            terminated,
            truncated,
        }
    }
}
