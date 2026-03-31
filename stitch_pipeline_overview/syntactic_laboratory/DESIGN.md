# Design System Document: The Precision Lab

## 1. Overview & Creative North Star
**The Creative North Star: "The Digital Curator"**
This design system moves away from the cluttered, "dashboard-in-a-box" aesthetic of typical data pipelines. Instead, it adopts the persona of a high-end scientific journal or a curated gallery. The goal is to elevate complex data into an editorial experience that feels authoritative yet effortless.

We achieve this through **Intentional Asymmetry** and **Tonal Depth**. By breaking the rigid 12-column grid and using overlapping elements or varied card widths, we create a rhythmic flow that guides the eye toward critical scientific insights. We prioritize white space not as "empty space," but as a functional component that reduces cognitive load in dense technical environments.

---

## 2. Colors & Surface Philosophy
The palette is rooted in `surface` (whites) and `surface_container` tiers (cool grays), providing a sterile, laboratory-grade foundation for `primary` (blue) and `tertiary` (teal) accents to signal protocol status.

### The "No-Line" Rule
**Standard 1px borders are strictly prohibited for sectioning.** 
Structural definition must be achieved through background shifts. For example, a main data feed sits on `surface`, while the contextual sidebar uses `surface_container_low`. This creates a sophisticated, "borderless" interface that feels infinite and modern.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of premium paper or frosted glass.
- **Base Layer:** `surface` (#f8f9fa)
- **Primary Layout Sections:** `surface_container_low` (#f1f4f6)
- **Interactive Cards/Nodes:** `surface_container_lowest` (#ffffff) to provide "lift."
- **Overlays/Modals:** `surface_bright` (#f8f9fa)

### The Glass & Gradient Rule
To prevent a "flat" academic look, use **Glassmorphism** for floating tooltips or navigation headers. Apply `surface_container_lowest` at 80% opacity with a `12px` backdrop-blur. 
- **Signature Accent:** Use a subtle linear gradient from `primary` (#1c5fa8) to `primary_container` (#d5e3ff) on the primary "Run Pipeline" CTA or active progress nodes to provide a sense of kinetic energy.

---

## 3. Typography: The Technical Editorial
We use a tri-font system to balance scientific precision with high-end legibility.

*   **Display & Headlines (Manrope):** Used for high-level metrics and page titles. Its geometric nature feels modern and engineered.
*   **Body & Titles (Inter):** The workhorse for technical data and labels. It ensures maximum readability at small scales.
*   **Labels (Space Grotesk):** Reserved for monospaced-style data points, version numbers, and system statuses. Its quirky terminals add a "scientific terminal" flavor without being unreadable.

**Hierarchy Strategy:**
*   `display-lg` is for hero metrics (e.g., total throughput).
*   `label-md` in `Space Grotesk` is used for status badges and protocol IDs to differentiate system-generated strings from human-readable text.

---

## 4. Elevation & Depth
Hierarchy is conveyed through **Tonal Layering**, not structural scaffolding.

*   **The Layering Principle:** A card (`surface_container_lowest`) should sit on a section (`surface_container_low`). The contrast is subtle (less than 2% shift), requiring the user's eye to perceive depth through color rather than lines.
*   **Ambient Shadows:** For floating elements (Modals/Popovers), use a shadow tinted with `on_surface` (#2b3437).
    *   *Spec:* `0 20px 40px rgba(43, 52, 55, 0.06)` — the blur must be wide and the opacity extremely low to mimic natural ambient light.
*   **The "Ghost Border":** If a container requires a border for accessibility, use `outline_variant` at **15% opacity**. This creates a hint of a boundary that disappears into the background upon quick glance.

---

## 5. Components

### Cards & Progress Nodes
*   **Cards:** Forbid divider lines. Use `spacing-6` (1.5rem) to separate header from content.
*   **Progress Nodes:** Represented as circles using `primary` or `tertiary`. Connect nodes with a `2px` path using `surface_variant`. Avoid arrows where possible; use the natural left-to-right flow and tonal "pulses" (gradients) to indicate direction.

### Buttons & Chips
*   **Primary Button:** Gradient fill (`primary` to `primary_dim`). Corner radius `md` (0.375rem).
*   **Action Chips:** Used for filtering data streams. Use `surface_container_high` for inactive states and `secondary_container` with `on_secondary_container` text for active states.
*   **Status Badges:** Use `tertiary_container` for "Success" and `error_container` for "Pipeline Breach." Text should always be the corresponding `on_container` token.

### Input Fields & Technical Data
*   **Inputs:** Use `surface_container_lowest` with a "Ghost Border." On focus, transition the border to `primary` at 100% opacity. 
*   **Data Lists:** Never use horizontal rules. Use zebra-striping with `surface_container_low` and `surface` to differentiate rows in massive datasets.

---

## 6. Do's and Don'ts

### Do
*   **Do** use `spacing-10` and `spacing-12` for large-scale layout gutters to create an "expensive" editorial feel.
*   **Do** overlap elements. For example, have a metric card slightly "break" the top boundary of a data table container.
*   **Do** use `label-sm` (Space Grotesk) for all "Metadata" to give it a distinct technical identity.

### Don't
*   **Don't** use pure black (#000) for text. Always use `on_surface` (#2b3437) to maintain the soft, academic aesthetic.
*   **Don't** use standard "Drop Shadows" on cards. Rely on background color shifts first.
*   **Don't** use icons as the primary way to convey meaning. In a scientific context, clear typography (`label-md`) is often more authoritative than a generic icon.
*   **Don't** use high-saturation reds. Use `error` (#9e3f4e) which is a more sophisticated, muted "Oxblood" red that feels professional, not alarming.