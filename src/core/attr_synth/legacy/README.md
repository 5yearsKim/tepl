# Legacy non-axis inference

These files preserve the former coordinate offset/stride matching and scalar
parameter inference for reference. They are excluded from all production build
targets and are never invoked by the compiler. They have no general completeness
guarantee.

The active `../axis_selectors.cc` matches axis selectors only. It may compare
supplied coordinate expressions to establish correspondence, but does not fill
coordinate or extent fields. Required non-axis attributes must be explicit;
schema defaults and optional `none` values remain available.

`construct.cc` preserves the former extent-domain assignment and deferred-domain
reconstruction path in its original context. Production construction now checks
output domains against supplied extent metadata instead.
