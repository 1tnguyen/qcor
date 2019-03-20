/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *******************************************************************************/
#ifndef COMPILER_QCORCOMPILER_HPP_
#define COMPILER_QCORCOMPILER_HPP_
#include "XACC.hpp"

using namespace xacc;

namespace qcor {

/**
 * The PyXACCCompiler is an XACC Compiler that compiles
 * python-like gate instruction source code to produce a
 * XACC IR.
 */
class QCORCompiler : public xacc::Compiler {

public:
  /**
   * The Compiler.
   */
  QCORCompiler() {}

  /**
   * Compile the given kernel code for the
   * given Accelerator.
   *
   * @param src The source code
   * @param acc Reference to the D-Wave Accelerator
   * @return
   */
  virtual std::shared_ptr<xacc::IR> compile(const std::string &src,
                                            std::shared_ptr<Accelerator> acc);

  /**
   * Compile the given kernel code.
   *
   * @param src The source code
   * @return
   */
  virtual std::shared_ptr<xacc::IR> compile(const std::string &src);

  const std::shared_ptr<Function>
  compile(std::shared_ptr<Function> f, std::shared_ptr<Accelerator> acc) override;
  
  /**
   * Return the command line options for this compiler
   *
   * @return options Description of command line options.
   */
  virtual std::shared_ptr<options_description> getOptions() {
    auto desc =
        std::make_shared<options_description>("QCOR Compiler Options");
    return desc;
  }

  virtual bool handleOptions(variables_map &map) { return false; }

  /**
   * We don't allow translations for the PyXACC Compiler.
   * @param bufferVariable
   * @param function
   * @return
   */
  virtual const std::string translate(const std::string &bufferVariable,
                                      std::shared_ptr<Function> function);

  virtual const std::string name() const { return "qcor"; }

  virtual const std::string description() const { return ""; }

  /**
   * The destructor
   */
  virtual ~QCORCompiler() {}
};

} // namespace xacc

#endif
