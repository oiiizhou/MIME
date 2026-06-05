# Natural Degradation Reference

This directory contains qualitative reference clips that illustrate naturally occurring degradation already present in MIME source videos before any controlled corruption is applied.

The clips are grouped by the dominant visible issue in each video:

- `foreground_or_object_occlusion`:
  Clips where foreground people, scene objects, vehicle structures, door frames, or other obstacles partially block the face or body of the target subject.
- `side_facing_or_turning_away`:
  Clips where the subject appears in a side-facing pose, turns away from the camera, or otherwise does not provide a clear frontal facial view.
- `low_light_or_low_visibility`:
  Clips affected by dark environments, weak illumination, low contrast, or other visibility-reducing conditions.
- `hand_or_body_occlusion`:
  Clips where hands, arms, or body parts cover important facial regions and reduce access to key expression cues.

Sample counts in this release:

- `foreground_or_object_occlusion`: 108 clips
- `side_facing_or_turning_away`: 59 clips
- `low_light_or_low_visibility`: 21 clips
- `hand_or_body_occlusion`: 5 clips

These reference clips are provided for qualitative inspection and rebuttal clarification. They are not defined as additional benchmark subsets and should not be interpreted as a separate evaluation split.
