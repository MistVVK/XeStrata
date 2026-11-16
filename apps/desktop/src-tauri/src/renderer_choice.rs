//! Linux: whether the window turns WebKitGTK's DMA-BUF renderer off (#5417).
//!
//! WebKitGTK's DMA-BUF renderer shows a blank or crashing window on several NVIDIA + Wayland desktops (tauri-apps/tauri#9394)
//! and on some virtual GPUs. `WEBKIT_DISABLE_DMABUF_RENDERER=1` avoids that, but it turns hardware acceleration off for the
//! whole window: every repaint is a full frame in shared memory, and on a 4K Radeon desktop typing and scrolling ran in slow
//! motion. So it is set only where it is needed: when the NVIDIA driver is loaded, or when a graphics device is driven by
//! anything other than the AMD and Intel drivers whose DMA-BUF path works (a virtual GPU, an unknown driver, nothing
//! readable). Anyone who set the variable themselves keeps their choice; that check stays at the caller.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::fs;
use std::path::Path;

/// Kernel drivers whose DMA-BUF renderer works in WebKitGTK.
const DMABUF_OK: &[&str] = &["amdgpu", "radeon", "i915", "xe"];

/// The decision from what the system shows: `nvidia` = the NVIDIA kernel module is loaded; `drivers` = the kernel driver of
/// each DRM card. Turn the DMA-BUF renderer off unless every card is driven by a driver it works with and NVIDIA is absent.
pub fn disable_dmabuf(nvidia: bool, drivers: &[String]) -> bool {
    nvidia || drivers.is_empty() || drivers.iter().any(|d| !DMABUF_OK.contains(&d.as_str()))
}

/// The DRM cards' kernel drivers under `sys` (`/sys` on a real system): the name the link `class/drm/cardN/device/driver`
/// points to. Connector entries (`card0-DP-1`) and render nodes are not cards.
pub fn drm_drivers(sys: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(sys.join("class/drm")) else { return out };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("card") else { continue };
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let driver = entry.path().join("device/driver");
        let target = fs::read_link(&driver).ok().and_then(|t| t.file_name().map(|n| n.to_string_lossy().into_owned()));
        // A card whose driver cannot be read counts as unknown, so the safe renderer stays.
        out.push(target.unwrap_or_default());
    }
    out.sort();
    out
}

/// The NVIDIA kernel module is loaded (`/sys/module/nvidia`).
pub fn nvidia_loaded(sys: &Path) -> bool {
    sys.join("module/nvidia").is_dir()
}

/// The whole decision for the system under `sys`.
pub fn disable_dmabuf_here(sys: &Path) -> bool {
    disable_dmabuf(nvidia_loaded(sys), &drm_drivers(sys))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn radeon_and_intel_keep_hardware_acceleration() {
        assert!(!disable_dmabuf(false, &v(&["amdgpu"])));
        assert!(!disable_dmabuf(false, &v(&["i915"])));
        assert!(!disable_dmabuf(false, &v(&["amdgpu", "i915"])));
        assert!(!disable_dmabuf(false, &v(&["xe"])));
    }

    #[test]
    fn nvidia_virtual_unknown_or_unreadable_keep_the_safe_renderer() {
        assert!(disable_dmabuf(true, &v(&["amdgpu"])), "a hybrid laptop with the NVIDIA driver loaded");
        assert!(disable_dmabuf(false, &v(&["nouveau"])));
        assert!(disable_dmabuf(false, &v(&["virtio-pci"])));
        assert!(disable_dmabuf(false, &v(&["vmwgfx"])));
        assert!(disable_dmabuf(false, &v(&["amdgpu", ""])), "one card without a readable driver");
        assert!(disable_dmabuf(false, &[]), "no card readable at all");
    }

    #[cfg(unix)]
    #[test]
    fn reads_the_cards_drivers_from_sysfs() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!("strata-drm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (card, driver) in [("card0", "amdgpu"), ("card1", "i915")] {
            let dev = root.join("class/drm").join(card).join("device");
            fs::create_dir_all(&dev).unwrap();
            fs::create_dir_all(root.join("bus/pci/drivers").join(driver)).unwrap();
            symlink(root.join("bus/pci/drivers").join(driver), dev.join("driver")).unwrap();
        }
        fs::create_dir_all(root.join("class/drm/card0-DP-1")).unwrap();
        fs::create_dir_all(root.join("class/drm/renderD128")).unwrap();
        assert_eq!(drm_drivers(&root), v(&["amdgpu", "i915"]));
        assert!(!disable_dmabuf_here(&root));
        fs::create_dir_all(root.join("module/nvidia")).unwrap();
        assert!(disable_dmabuf_here(&root));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unreadable_sysfs_keeps_the_safe_renderer() {
        let root = std::env::temp_dir().join(format!("strata-drm-none-{}", std::process::id()));
        assert!(drm_drivers(&root).is_empty());
        assert!(disable_dmabuf_here(&root));
    }
}
