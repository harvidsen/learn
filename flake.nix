{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:

    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {

      devShells.x86_64-linux.pyomo = pkgs.mkShell {
        buildInputs = [
          pkgs.python3Packages.pyomo
        ];
      };

    };
}
