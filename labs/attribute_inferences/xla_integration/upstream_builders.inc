absl::StatusOr<std::unique_ptr<HloModule>> make_FuseDotWithConstantOperands(){

  auto builder = HloComputation::Builder("entry");
  // (1.0 + 2.0) * (2.0 - 3.0)
  HloInstruction* const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* const2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  HloInstruction* const3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      const1->shape(), HloOpcode::kAdd, const1, const2));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      const2->shape(), HloOpcode::kSubtract, const2, const3));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      add->shape(), HloOpcode::kMultiply, add, sub));

  auto module = CreateNewVerifiedModule("fuse_with_constant_operands");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(mul));
  HloInstruction* call = module->OutlineExpressionFromComputation(
      {add, sub, mul}, "entry", entry_computation);
  
 (void)entry_computation;
 (void)call;
return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> make_FoldConvDimSwapTransposeRhs(){

  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_y =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), y, {1, 0, 2, 3}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(
        transpose_y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  ABSL_ASSIGN_OR_RETURN(
      const Shape conv_shape,
      ShapeInference::InferConvolveShape(
          x->shape(), transpose_y->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums, SparsityConfig(),
          /*preferred_element_type=*/std::nullopt));
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, {x, transpose_y},
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  
 (void)entry_computation;
return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> make_FoldConvComplexTransposeRhs(){

  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {1, 2, 1, 3}),
      /*name=*/"y"));
  HloInstruction* transpose_y =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), y, {1, 3, 0, 2}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(
        transpose_y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  ABSL_ASSIGN_OR_RETURN(
      const Shape conv_shape,
      ShapeInference::InferConvolveShape(
          x->shape(), transpose_y->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums, SparsityConfig(),
          /*preferred_element_type=*/std::nullopt));
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, {x, transpose_y},
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  
 (void)entry_computation;
return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> make_FoldConvTransposeLhs(){

  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_x =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), x, {1, 0, 2, 3}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  ABSL_ASSIGN_OR_RETURN(
      const Shape conv_shape,
      ShapeInference::InferConvolveShape(
          transpose_x->shape(), y->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*sparsity_config=*/SparsityConfig(),
          /*preferred_element_type=*/std::nullopt));
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, {transpose_x, y},
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  
 (void)entry_computation;
return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> make_FoldConvComplexTransposeLhs(){

  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_x =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), x, {1, 0, 3, 2}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  ABSL_ASSIGN_OR_RETURN(
      const Shape conv_shape,
      ShapeInference::InferConvolveShape(
          transpose_x->shape(), y->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*sparsity_config=*/SparsityConfig(),
          /*preferred_element_type=*/std::nullopt));
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, {transpose_x, y},
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  
 (void)entry_computation;
return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> makeBuilder(const std::string& name){
if(name=="FuseDotWithConstantOperands")return make_FuseDotWithConstantOperands();
if(name=="FoldConvDimSwapTransposeRhs")return make_FoldConvDimSwapTransposeRhs();
if(name=="FoldConvComplexTransposeRhs")return make_FoldConvComplexTransposeRhs();
if(name=="FoldConvTransposeLhs")return make_FoldConvTransposeLhs();
if(name=="FoldConvComplexTransposeLhs")return make_FoldConvComplexTransposeLhs();
return absl::InvalidArgumentError("unknown builder");
}
