/**
 * OrthoVision AI design tokens — dark clinical theme.
 * Radiographs are near-grayscale, so the UI stays deep navy with a teal
 * accent and reserves saturated hues for segmentation classes.
 */
import { Platform } from 'react-native';

export const colors = {
  bg: '#0A1118',
  bgElevated: '#101B26',
  surface: 'rgba(148, 197, 233, 0.06)',
  surfaceBorder: 'rgba(148, 197, 233, 0.14)',
  primary: '#2DD4BF',
  primaryDim: 'rgba(45, 212, 191, 0.16)',
  text: '#F1F7FB',
  textSecondary: '#8FA3B5',
  textTertiary: '#5C7186',
  danger: '#F87171',
  online: '#34D399',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  card: 24,
  control: 16,
  pill: 999,
} as const;

export const fonts = Platform.select({
  ios: { sans: 'system-ui', rounded: 'ui-rounded', mono: 'ui-monospace' },
  default: { sans: 'normal', rounded: 'normal', mono: 'monospace' },
});
