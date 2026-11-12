import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // The DOM lookups in main.ts use `!` for elements that index.html always has.
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },
);
