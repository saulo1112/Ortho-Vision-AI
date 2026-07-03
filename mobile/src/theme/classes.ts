/** Presentation config for each segmentation class returned by the API. */

export type ImplantClassName =
  | 'clavo_intramedular'
  | 'placa_atornillada'
  | 'protesis_articular';

export interface ClassStyle {
  /** English display label (raw class_name stays Spanish in the API). */
  label: string;
  color: string;
}

export const CLASS_STYLES: Record<string, ClassStyle> = {
  clavo_intramedular: { label: 'Intramedullary nail', color: '#38BDF8' },
  placa_atornillada: { label: 'Screwed plate', color: '#34D399' },
  protesis_articular: { label: 'Joint prosthesis', color: '#FBBF24' },
};

export function classStyle(className: string): ClassStyle {
  return CLASS_STYLES[className] ?? { label: className, color: '#A78BFA' };
}
