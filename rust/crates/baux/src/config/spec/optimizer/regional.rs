use super::*;

/// Policies that turn one Bayesian search into concurrent regional searches.
///
/// Surrogate choice, acquisition, observation sharing, budget allocation,
/// controller updates, and restart memory remain independent decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct RegionalSearchSpec {
    pub count: usize,
    #[deser(default)]
    pub sharing: ObservationSharing,
    #[deser(default)]
    pub allocation: RegionAllocation,
    #[deser(default)]
    pub controller: RegionController,
    #[deser(default = region_successes())]
    pub success_tolerance: i32,
    #[deser(default = region_failures())]
    pub failure_tolerance: i32,
    #[deser(default)]
    pub restart_memory: RegionMemory,
    #[deser(default = restart_candidates())]
    pub restart_candidates: usize,
    #[deser(default)]
    pub restart_center: RestartCenter,
    #[deser(default)]
    pub acquisitions: Vec<Option<AcquisitionSpec>>,
}

impl RegionalSearchSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.count == 0 {
            return Err("regional search count must be positive".into());
        }
        if self.success_tolerance <= 0 || self.failure_tolerance <= 0 {
            return Err("regional search tolerances must be positive".into());
        }
        if self.restart_candidates == 0 {
            return Err("regional restart candidates must be positive".into());
        }
        if !self.acquisitions.is_empty() && self.acquisitions.len() != self.count {
            return Err("regional acquisitions must be empty or match region count".into());
        }
        Ok(())
    }

    pub(crate) fn runtime(
        &self,
        bounds: &TrustRegionBounds,
    ) -> Result<crate::optimizer::MultiTrustRegionConfig, String> {
        self.validate()?;
        bounds.validate()?;
        use crate::optimizer::{
            MultiTrustRegionConfig, RegionController as RuntimeController, RestartCenterPolicy,
            RestartCenterSelection, RestartMemoryPolicy,
        };
        let mut config = MultiTrustRegionConfig::new(
            self.count,
            crate::trust_region::TRLengthConfig::new(bounds.initial, bounds.min, bounds.max),
        );
        config.sharing_policy = self.sharing.runtime();
        config.allocation_policy = self.allocation.runtime();
        config.controller = match self.controller {
            RegionController::Turbo => RuntimeController::Turbo,
            RegionController::FixedRadius => RuntimeController::FixedRadius,
        };
        config.succ_tolerance = self.success_tolerance;
        config.fail_tolerance = self.failure_tolerance;
        config.restart_memory_policy = match self.restart_memory {
            RegionMemory::Fresh => RestartMemoryPolicy::Fresh,
            RegionMemory::RetainRelevant => RestartMemoryPolicy::RetainRelevant,
        };
        config.restart_center_policy = RestartCenterPolicy {
            candidates: self.restart_candidates,
            selection: match self.restart_center {
                RestartCenter::First => RestartCenterSelection::First,
                RestartCenter::Maximin => RestartCenterSelection::Maximin,
            },
        };
        config.region_acquisitions = if self.acquisitions.is_empty() {
            vec![None; self.count]
        } else {
            self.acquisitions
                .iter()
                .map(|item| item.as_ref().map(AcquisitionConfig::from_spec))
                .collect()
        };
        Ok(config)
    }
}

impl ObservationSharing {
    fn runtime(self) -> crate::optimizer::SharingPolicy {
        use crate::optimizer::SharingPolicy;
        match self {
            Self::BoxContain => SharingPolicy::BoxContain,
            Self::EllipsoidBall => SharingPolicy::EllipsoidBall,
            Self::NearestCenter => SharingPolicy::NearestCenter,
            Self::OriginOnly => SharingPolicy::OriginOnly,
        }
    }
}

impl RegionAllocation {
    fn runtime(self) -> crate::optimizer::AllocationPolicy {
        use crate::optimizer::AllocationPolicy;
        match self {
            Self::IncumbentUcb1 => AllocationPolicy::IncumbentUcb1,
            Self::LeastTrials => AllocationPolicy::LeastTrials,
            Self::JointAcquisition => AllocationPolicy::JointAcquisition,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObservationSharing {
    #[default]
    BoxContain,
    EllipsoidBall,
    NearestCenter,
    OriginOnly,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum RegionAllocation {
    #[default]
    IncumbentUcb1,
    LeastTrials,
    JointAcquisition,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum RegionController {
    #[default]
    Turbo,
    FixedRadius,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum RegionMemory {
    #[default]
    Fresh,
    RetainRelevant,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum RestartCenter {
    #[default]
    First,
    Maximin,
}

const fn region_successes() -> i32 {
    3
}
const fn region_failures() -> i32 {
    5
}
const fn restart_candidates() -> usize {
    1
}
