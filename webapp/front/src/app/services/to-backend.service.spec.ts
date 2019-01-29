import { TestBed } from '@angular/core/testing';

import { ToBackendService } from './to-backend.service';

describe('ToBackendService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: ToBackendService = TestBed.get(ToBackendService);
    expect(service).toBeTruthy();
  });
});
