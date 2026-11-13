"""Pinned branch substitutions. Guards and commit APIs remain owned by XLA."""
import re

MARKER = "// TEPL branch adapter v2"
PORTS = {
    934: ("add", "lhs, rhs", "new"),
    1405: ("logical_and", "lhs, rhs", "compatible"),
    2311: ("constant", "constant", "add"),
    2317: ("constant", "constant", "new"),
    2478: ("divide", "a, b", "compatible"),
    2824: ("transpose", "broadcast->mutable_operand(0)", "add"),
    5025: ("multiply", "lhs, rhs", "compatible"),
    5056: ("multiply", "a, b", "inplace"),
    5389: ("get_tuple_element", "get_tuple_element", "compatible"),
    5605: ("broadcast", "operand", "new"),
    5690: ("broadcast", "operand", "new"),
    5723: ("broadcast", "operand", "new"),
    5761: ("compare", "lhs, rhs", "add"),
    5767: ("compare", "lhs, rhs", "add"),
    6744: ("reshape", "operand", "existing"),
    6750: ("reshape", "operand->mutable_operand(0)", "new"),
    8661: ("reduce", "init_for_single_output", "new"),
    9835: ("select", "select->mutable_operand(0), select->mutable_operand(1), select->mutable_operand(2)", "compatible"),
    9840: ("select", "select->mutable_operand(0), select->mutable_operand(1), select->mutable_operand(2)", "compatible"),
    5132: ("multiply", "a, c1, c2", "new"),
    5165: ("multiply", "a, op->mutable_operand(0), constant, op", "new"),
    2614: ("divide", "a, b", "new"),
    2590: ("divide", "a, c, b", "add"),
    5645: ("broadcast", "operand->mutable_operand(0)", "new"),
    5681: ("user", "operand", "uses"),
    6358: ("user", "", "loop"),
    10187: ("transpose", "rhs, lhs", "dot"),
    2742: ("dot", "dot->mutable_operand(0), dot->mutable_operand(1)", "dot_units"),
    3785: ("dot", "lhs, rhs", "new"),
    6898: ("reshape", "operand->mutable_operand(0)", "new"),
    6945: ("reshape", "iota", "add"),
    8870: ("reduce", "negate_arg, init_value", "reduce_negate"),
    9023: ("reduce", "arg->mutable_operand(0), init_value", "new"),
    9050: ("reduce", "arg->mutable_operand(0), init_value", "add"),
    10505: ("convolution", "a, b", "conv"),
}


def invocation(site):
    root, captures, _ = PORTS[site]
    return f'''ABSL_ASSIGN_OR_RETURN(HloInstruction* tepl_replacement,
        TryTeplBranch({root}, {site}, {{{captures}}}{', reshape_degenerate->inserted_dimensions' if site == 5645 else ''}));'''


def commit(site):
    root, _, kind = PORTS[site]
    if kind == "new":
        flags = ', /*preserve_sharding=*/false, /*relay_control_dependency=*/true' if site == 6750 else ''
        call = f"ReplaceWithNewInstruction({root}, std::unique_ptr<HloInstruction>(tepl_replacement){flags})"
    else:
        flags = ', /*preserve_frontend_attributes=*/false' if site == 6744 else ''
        if kind == "add":
            value = f"{root}->AddInstruction(std::unique_ptr<HloInstruction>(tepl_replacement))"
        else:
            value = "tepl_replacement"
        call = f"ReplaceInstruction({root}, {value}{flags})"
    return f"return ObserveAlgebraicRewrite({site}, __func__, [&]() {{ return {call}; }});"


def apply(source):
    if MARKER in source:
        return source
    block = '''// TEPL branch adapter v2
using TeplBranchHook = absl::StatusOr<HloInstruction*> (*)(
    HloInstruction*, int, absl::Span<HloInstruction* const>,
    absl::Span<const int64_t>, absl::Span<const int64_t>);
namespace { TeplBranchHook tepl_branch_hook = nullptr; }
void SetTeplBranchHook(TeplBranchHook hook) { tepl_branch_hook = hook; }
absl::StatusOr<HloInstruction*> TryTeplBranch(
    HloInstruction* root, int site, absl::Span<HloInstruction* const> captures,
    absl::Span<const int64_t> first = {}, absl::Span<const int64_t> second = {}) {
  return tepl_branch_hook(root, site, captures, first, second);
}

'''
    source = source.replace("namespace xla {", "namespace xla {\n" + block, 1)
    for site, (root, _, kind) in PORTS.items():
        if site in {2742,3785,6898,6945,8870,9023,9050,10505}:
            anchors = {
                2742: '  HloInstruction* new_lhs =\n      num_degenerate_lhs_dims > 0',
                3785: '  ABSL_ASSIGN_OR_RETURN(HloInstruction * new_lhs,\n                        NormalizeDotOperandToBatchMajorAndContractingMinor(',
                6898: '      return ObserveAlgebraicRewrite(6898,',
                6945: '      int64_t multiplier = 1;',
                8870: '    ABSL_RETURN_IF_ERROR(reduce->ReplaceOperandWith(0, negate_arg));',
                9023: '      ObserveAlgebraicRewrite(9009,',
                9050: '      absl::InlinedVector<int64_t, 8> new_reduce_dims;',
                10505: '    auto new_conv =\n        convolution->CloneWithNewOperands(convolution->shape(), {a, b});',
            }
            anchor = anchors[site]
            call = invocation(site)
            prefix = ''
            action = commit(site)
            if site in {6898,9023,9050}:
                pairs = ('ShapeUtil::DimensionsUnmodifiedByReshape(operand->shape(), reshape->shape())'
                         if site == 6898 else 'unmodified_dims')
                prefix = f'''std::vector<int64_t> tepl_from, tepl_to;
      for (auto pair : {pairs}) {{ tepl_from.push_back(pair.first); tepl_to.push_back(pair.second); }}
'''
                call = call.replace('}));','}, tepl_from, tepl_to));')
            if site == 6945:
                call = call.replace('}));','}, {iota_dim->second, (iota_dim+1)->second}));')
            if site == 2742:
                action = '''if (tepl_replacement->opcode() == HloOpcode::kDot) {
          ABSL_RETURN_IF_ERROR(ObserveAlgebraicRewrite(2742, __func__, [&]() {
            return ReplaceInstruction(dot, dot->AddInstruction(std::unique_ptr<HloInstruction>(tepl_replacement)));
          }));
        } else {
          ABSL_RETURN_IF_ERROR(ObserveAlgebraicRewrite(2744, __func__, [&]() {
            return ReplaceWithNewInstruction(dot, std::unique_ptr<HloInstruction>(tepl_replacement),
                 /*preserve_sharding=*/false, /*relay_control_dependency=*/true);
          }));
        } return true;'''
            if site == 8870:
                prefix = 'auto tepl_old_users = reduce->users();\n'
                action = '''auto* tepl_negate = reduce->AddInstruction(std::unique_ptr<HloInstruction>(tepl_replacement));
        ObserveAlgebraicRewrite(8870, __func__, [&]() { return MarkAsChanged(); });
        return reduce->ReplaceUsesWith(tepl_old_users, tepl_negate);'''
            if site == 9023:
                action = 'ObserveAlgebraicRewrite(9009, __func__, [&]() {return MarkAsChanged();});\n' + action
            if site == 10505:
                action = '''ABSL_RETURN_IF_ERROR(ObserveAlgebraicRewrite(10505, __func__, [&]() {
          return ReplaceWithNewInstruction(convolution, std::unique_ptr<HloInstruction>(tepl_replacement));
        })); return true;'''
            addition = f'''    if (tepl_branch_hook) {{
      {prefix}{call}
      if (tepl_replacement != nullptr) {{ {action} }}
    }}
'''
            boundary = source.index(f"ObserveAlgebraicRewrite({site},")
            position = source.rfind(anchor,0,boundary)
            if position < 0:
                # The endpoint itself is the safe insertion boundary.
                position = source.find(anchor)
            if position < 0: raise ValueError(f"missing branch boundary {site}")
            source = source[:position] + addition + source[position:]
            continue
        if site in {5132, 5165, 2614, 2590, 5681, 6358, 10187}:
            anchors = {
                5132: '      ABSL_ASSIGN_OR_RETURN(auto* product_of_constants,',
                5165: '        auto dims = op->dimensions();',
                2614: '    auto* recip = bcast->AddInstruction(HloInstruction::CreateBinary(',
                2590: '            ABSL_RETURN_IF_ERROR(InvertConstant<NativeT>(*c, &new_literal));',
                5681: '        HloInstruction* new_broadcast =',
                6358: '    std::vector<HloInstruction*> new_operands;',
                10187: '    DotDimensionNumbers new_dnums = dnums;',
            }
            anchor = anchors[site]
            if site == 6358:
                call = '''ABSL_ASSIGN_OR_RETURN(HloInstruction* tepl_replacement,
        TryTeplBranch(user, 6358, {
          user->mutable_operand(0)->mutable_operand(0),
          user->mutable_operand(std::min<int64_t>(1, user->operand_count()-1))->mutable_operand(0),
          user->mutable_operand(std::min<int64_t>(2, user->operand_count()-1))->mutable_operand(0), broadcast}));'''
                action = '''ABSL_RETURN_IF_ERROR(ObserveAlgebraicRewrite(6358, __func__, [&]() {
          return user->ReplaceAllUsesWith(user->AddInstruction(std::unique_ptr<HloInstruction>(tepl_replacement)));
        })); changed = true; continue;'''
            elif site == 5681:
                call = invocation(site)
                action = '''ObserveAlgebraicRewrite(5681, __func__, [&]() {return MarkAsChanged();});
          return user->ReplaceAllUsesWith(user->AddInstruction(std::unique_ptr<HloInstruction>(tepl_replacement)));'''
            elif site == 10187:
                call = invocation(site)
                action = '''ABSL_RETURN_IF_ERROR(ObserveAlgebraicRewrite(10187, __func__, [&]() {
          return ReplaceWithNewInstruction(transpose, std::unique_ptr<HloInstruction>(tepl_replacement));
        })); return true;'''
            else:
                call, action = invocation(site), commit(site)
            addition = f'''    if (tepl_branch_hook) {{
      {call}
      if (tepl_replacement != nullptr) {{ {action} }}
    }}
'''
            boundary = source.index(f"ObserveAlgebraicRewrite({site},")
            position = source.rfind(anchor, 0, boundary)
            if position < 0:
                raise ValueError(f"missing branch boundary {site}")
            source = source[:position] + addition + source[position:]
            continue
        if kind == "compatible":
            if site == 5389:
                source = source.replace("if (ObserveAlgebraicRewrite(5389,", "if (true && ObserveAlgebraicRewrite(5389,", 1)
            # Expand the short-circuit if. The literal/type predicate is still
            # evaluated before calling TEPL, and the original compatibility
            # helper remains responsible for sharding and shape policy.
            pattern = (r"if \(([^{};]*?) &&\s*ObserveAlgebraicRewrite\(" + str(site) +
                       r", __func__, \[&\]\(\) \{ return ReplaceInstructionIfCompatible\((.*?)\); \}\)\) \{\s*return absl::OkStatus\(\);\s*\}")
            matches = list(re.finditer(pattern, source, re.S))
            if len(matches) != 1:
                raise ValueError(f"missing compatible branch {site}")
            match = matches[0]
            old_args = match[2].strip()
            # Split only at the first comma (the second expression may call
            # mutable_operand). Parentheses remain untouched.
            old_target = old_args.split(",", 1)[1].strip()
            replacement = f'''if ({match[1]}) {{
      HloInstruction* tepl_target = {old_target};
      if (tepl_branch_hook) {{
        {invocation(site)}
        if (tepl_replacement != nullptr) tepl_target = tepl_replacement;
      }}
      if (ObserveAlgebraicRewrite({site}, __func__, [&]() {{
            return ReplaceInstructionIfCompatible({root}, tepl_target);
          }})) return absl::OkStatus();
    }}'''
            source = source[:match.start()] + replacement + source[match.end():]
            continue
        if kind == "inplace":
            anchor = "      ABSL_RETURN_IF_ERROR(multiply->ReplaceOperandWith(0, a));"
            addition = f'''      if (tepl_branch_hook) {{
        {invocation(site)}
        if (tepl_replacement != nullptr) {{
          std::unique_ptr<HloInstruction> owned(tepl_replacement);
          ABSL_RETURN_IF_ERROR(multiply->ReplaceOperandWith(0, owned->mutable_operand(0)));
          ABSL_RETURN_IF_ERROR(multiply->ReplaceOperandWith(1, owned->mutable_operand(1)));
          ObserveAlgebraicRewrite({site}, __func__, [&]() {{ return MarkAsChanged(); }});
          return absl::OkStatus();
        }}
      }}
'''
        else:
            if site == 2311:
                anchor = "    Literal unique_scalar("
            elif site == 5723:
                anchor = "    auto new_operand = operand->AddInstruction(HloInstruction::CreateReshape("
            else:
                pattern = r"^[ \t]*return ObserveAlgebraicRewrite\(" + str(site) + r", __func__,"
                matches = list(re.finditer(pattern, source, re.M))
                if len(matches) != 1:
                    raise ValueError(f"missing replacement branch {site}")
                anchor = matches[0][0]
            addition = f'''    if (tepl_branch_hook) {{
      {invocation(site)}
      if (tepl_replacement != nullptr) {{ {commit(site)} }}
    }}
'''
        if source.count(anchor) != 1:
            raise ValueError(f"ambiguous branch boundary {site}")
        source = source.replace(anchor, addition + anchor, 1)
    return source
