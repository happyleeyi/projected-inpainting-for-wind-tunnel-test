## TPU Tap Mapping Notes

Source: dataset `Readme` variable from the TPU `.mat` files.

- `Location_of_measured_points` is `[4, N]` and each column is one measured point (tap).
- Mapping by row index:
  - `loc[0]`: left distance.
  - `loc[1]`: bottom distance.
  - `loc[2]`: point id.
  - `loc[3]`: face id.

Surface mapping:
- `1`: windward
- `2`: right sideward
- `3`: leeward
- `4`: left sideward

Origin:
- Left-bottom of the windward surface.
