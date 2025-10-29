{
  description = "Hello world flake using uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      config.allowUnfree = true;
      config.cudaSupport = true;
      config.cudaVersion = "12";

      # Load a uv workspace from a workspace root.
      # Uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create package overlay from workspace.
      overlay = workspace.mkPyprojectOverlay {
        # Prefer prebuilt binary wheels as a package source.
        # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
        # Binary wheels are more likely to, but may still require overrides for library dependencies.
        sourcePreference = "wheel"; # or sourcePreference = "sdist";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups
      #
      # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
      # This is an additional overlay implementing build fixups.
      # See:
      # - https://pyproject-nix.github.io/uv2nix/FAQ.html
      pyprojectOverrides = _final: _prev: {
        # Implement build fixups here.
        # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
        # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
      };

      # This example is only using x86_64-linux
      pkgs = nixpkgs.legacyPackages.x86_64-linux;

      # Use Python 3.12 from nixpkgs
      python = pkgs.python312;

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );

    in
    {
      # Package a virtual environment as our main application.
      #
      # Enable no optional dependencies for production build.
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "timesfm" workspace.deps.default;

      # Make hello runnable with `nix run`
      apps.x86_64-linux = {
        default = {
          type = "app";
          program = "${self.packages.x86_64-linux.default}/bin/hello";
        };
      };

      # This example provides two different modes of development:
      # - Impurely using uv to manage virtual environments
      # - Pure development using uv2nix to manage virtual environments
      devShells.x86_64-linux = {
        # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
        # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
        uv-venv = pkgs.mkShell {
          name = "timesfm";
          packages = [
            python
            pkgs.uv
            pkgs.fish
          ];
          env = {
            # Prevent uv from managing Python downloads
            UV_PYTHON_DOWNLOADS = "never";
            # Force uv to use nixpkgs Python interpreter
            UV_PYTHON = python.interpreter;
          }
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
            LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };
          shellHook = ''
            echo "Welcome to the file-to-csv development shell!"
            echo "Using Python: $(which python)"
            echo "Using uv: $(which uv)"
            echo "Using fish: $(which fish)"
            unset PYTHONPATH
          '';
        };

        # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
        # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
        #
        # This means that any changes done to your local files do not require a rebuild.
        #
        # Note: Editable package support is still unstable and subject to change.
        default =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "hello-world" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope (
              lib.composeManyExtensions [
                editableOverlay
                # Apply fixups for building an editable package of your workspace packages
                (final: prev: {
                  "nvidia-cufile-cu12" = prev."nvidia-cufile-cu12".overrideAttrs (old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core ];
                  });

                  "nvidia-cusolver-cu12" = prev."nvidia-cusolver-cu12".overrideAttrs (old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudatoolkit
                      pkgs.linuxPackages.nvidia_x11
                    ];
                  });

                  "nvidia-cusparse-cu12" = prev."nvidia-cusparse-cu12".overrideAttrs (old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudatoolkit
                      pkgs.linuxPackages.nvidia_x11
                    ];
                  });

                  "nvidia-nvshmem-cu12" = prev."nvidia-nvshmem-cu12".overrideAttrs (old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [
                      pkgs.rdma-core
                      pkgs.libGL
                      pkgs.libGLU
                      pkgs.pmix
                      pkgs.libfabric
                      pkgs.mpi
                    ];
                  });

                  timesfm = prev.timesfm.overrideAttrs (old: {
                    # It's a good idea to filter the sources going into an editable build
                    # so the editable package doesn't have to be rebuilt on every change.
                    src = lib.fileset.toSource {
                      root = old.src;
                      fileset = lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                        (old.src + "/src")
                      ];
                    };

                    # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                    #
                    # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                    # This behaviour is documented in PEP-660.
                    #
                    # With Nix the dependency needs to be explicitly declared.
                    nativeBuildInputs =
                      old.nativeBuildInputs
                      ++ final.resolveBuildSystem {
                        editables = [ ];
                      };
                  });

                })
                (final: prev: {
                  # do a separate overlay function so we can rely on the output of the first one?
                  "torch" = prev."torch".overrideAttrs (old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [
                      pkgs.rdma-core
                      pkgs.cudaPackages.cuda_cudart
                      pkgs.cudaPackages.libcufile
                      pkgs.cudaPackages.libcusparse
                      pkgs.cudaPackages.cusparselt
                      pkgs.cudaPackages.nccl
                      pkgs.cudatoolkit
                      pkgs.linuxPackages.nvidia_x11
                      pkgs.libGL
                      pkgs.libGLU
                      pkgs.pmix
                      pkgs.libfabric
                      pkgs.mpi
                      pkgs.cudaPackages.cudnn
                      final."nvidia-nvshmem-cu12"
                    ];
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                      pkgs.autoPatchelfHook
                    ];
                    postFixup = ''
                      addAutoPatchelfSearchPath ${final."nvidia-nvshmem-cu12"}/lib
                    '';
                  });
                })
              ]
            );

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "timesfm" workspace.deps.all;

          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.fmt.dev
              pkgs.cudaPackages.cuda_cudart
              pkgs.ffmpeg
              pkgs.fmt.dev
              pkgs.cudatoolkit
              pkgs.linuxPackages.nvidia_x11
              pkgs.cudaPackages.cudnn
              pkgs.libGLU
              pkgs.libGL
              pkgs.xorg.libXi
              pkgs.xorg.libXmu
              pkgs.freeglut
              pkgs.xorg.libXext
              pkgs.xorg.libX11
              pkgs.xorg.libXv
              pkgs.xorg.libXrandr
              pkgs.zlib
              pkgs.ncurses5
              pkgs.stdenv.cc
              pkgs.binutils
              pkgs.redis
            ];

            env = {
              # Don't create venv using uv
              UV_NO_SYNC = "1";

              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;

              # Prevent uv from downloading managed Python's
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH

              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              alias redis-start='redis-server redis.conf &'
            '';
          };
      };
    };
}
